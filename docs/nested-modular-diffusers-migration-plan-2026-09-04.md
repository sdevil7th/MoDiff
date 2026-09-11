# Nested Modular Diffusers Migration Plan

Date: 2026-09-04

Implementation status: recursive hierarchy, progressive disclosure, and the
shared internal Block projection are implemented. Per-model execution and
publication qualification continue as a separate evidence campaign.

## Outcome

Migrate every registered Diffusers Cluster and every upstream Modular Diffusers
block used by those Clusters to one recursive `BlockDefinitionV2` experience:

- a registered Cluster remains the immutable, collapsed one-node pipeline;
- expanding it reveals the active upstream Modular Diffusers hierarchy rather
  than one flattened row of children;
- every sequential, conditional, auto, loop, and leaf block is an ordinary
  selectable, movable, resizable, connectable canvas node;
- every upstream block is also searchable and draggable from the left panel
  under **Modular Diffusers Blocks**;
- edits are copy-on-write workflow-instance data and never mutate a registered
  catalog definition;
- selecting any hierarchy level offers **Save as new User Node**; an existing
  user-owned definition additionally offers **Update existing User Node**;
- save, browser refresh, backend restart, collapse, and re-expansion preserve
  exact prompts, parameters, component/model choices, topology, interfaces,
  hierarchy expansion, sizes, and positions.

No alternate executor or client-owned pipeline format is introduced. The
backend graph remains execution authority, and modified Modular workflows are
rebuilt through the pinned Diffusers `init_pipeline()` path.

## Audited upstream and application facts

The pinned unpruned contract covers 34 Modular pipeline classes and 94 public
workflows. It contains 632 exact block definitions: 598 definitions placed in
the upstream trees and 34 pipeline roots. The deepest upstream placement is
five Modular levels beneath the pipeline root, so the deepest visible tree is
six levels when the outer registered Cluster is counted.

The current selected-execution snapshot contains 483 exact definitions, but
`get_execution_blocks()` resolves conditionals and flattens some selected
container names. That snapshot remains useful for execution admission; it is
not sufficient as the visual hierarchy source. The unpruned conditional
snapshot plus its workflow execution traces are the hierarchy authority.

MoDiff Client uses `@xyflow/react` 12.8.4. React Flow supports recursive
subflows through chained `parentId` relationships, relative positions,
`extent`, and `expandParent`. Current MoDiff V2 projection assigns every child
directly to the outer Block and therefore discards its already-persisted
`parentPlacementPath` relationship.

Current catalog discovery also marks every Modular Diffusers block
`insertable: false`. Discoverability is not sufficient for modular editing.

## Terminology and invariants

### Registered Cluster

The immutable first-party full workflow. Registration controls provenance,
execution admission, publication, and Auto authority. It does not select a
different renderer.

### Modular Diffusers Block

One exact upstream container or leaf definition. A container may expose its
children recursively. It is not a nested registered Cluster and it is not a
nested User Node.

### User Node

A user-owned `BlockDefinitionV2` materialized from the current state of a full
Cluster, a Modular container subtree, a loop, a leaf, or an ordinary selected
subgraph. It retains provenance but cannot claim registered ownership.

Nested User Node or Cluster references remain prohibited. Saving a subtree
copies and rebases its semantic graph into a new independent top-level User
Node; it never stores a live reference to the containing Cluster.

## Contract strategy

`BlockDefinitionV2.graph` remains the canonical flat execution graph. Exact
Modular containment is already hash-covered by
`BlockGraphNodeV2.modularDiffusers.placementPath` and
`parentPlacementPath`. Recursive React Flow parents are a projection derived
from those fields, not a second execution graph.

Every upstream placement is rendered through the same Block V2 frame whether
it is the registered root, a structural container, an executable container,
or a leaf. Internal frames are projections, not nested persistence
authorities: changing a control, moving a child, deleting/replacing a subtree,
or reconnecting an edge writes to the one owning workflow
`BlockInstanceV2.effectiveGraph`/`values`/`presentation`. Saving an internal
frame as a User Node materializes a new independent top-level definition.

Fresh hierarchical instances seed `presentation.collapsedContainerNodeIds`
with every upstream placement that owns children. Expanding the outer Cluster
therefore reveals only its immediate internal Blocks. Expanding one of those
reveals the next level while deeper descendants remain collapsed. The array is
presentation-only, persists with the workflow, and may name executable
`custom` nodes as well as structural `group` nodes because upstream ownership,
not a renderer type, defines a Modular container.

An upstream node copied from the left-panel catalog additionally retains the
three-part immutable origin `sourceDefinitionId`, `sourcePlacementPath`, and
`sourceExecutionScope`. Its `placementPath` may change when the workflow owner
moves or replaces that instance; the source triplet never changes. This is how
the client can lower an edited canvas back to an exact pinned insert/replace
recipe without guessing from a class name.

`BlockInstanceV2.presentation` gains only presentation data required for
recursive editing:

- expansion state keyed by stable semantic container node ID;
- layout entries interpreted relative to each node's immediate semantic
  parent;
- deterministic fallbacks for definitions saved before recursive projection.

`internalLayout` records a container's compact own box, not the recursively
expanded dimensions of its descendants. Expanded bounds are a projection:
visible children are measured bottom-up, each ancestor receives explicit
header/right/bottom padding, and colliding siblings are shifted without
changing their durable preferred coordinates. A late control or connector
measurement must rematerialize the complete ancestor chain in one store update.
Collapse therefore restores the compact box and can never preserve a stale
expanded envelope or clip descendants after a later re-expansion. Legacy
entries whose stored dimensions already contain their children are normalized
to compact own bounds during projection.

Any contract addition must update the client and backend validators,
canonicalization where execution identity is affected, API documentation,
cross-runtime fixtures, size bounds, and legacy normalization in the same
change. Presentation-only fields do not enter definition content hashes.

Subtree sockets follow the same rule in both collapsed and expanded states. A
visible internal container derives typed input/output handles only for public
bindings and semantic edges that cross that subtree boundary. When expanded,
the exact internal edge remains attached to the visible leaf, while the
container retains its interface for ordinary external connections. Those
canvas handles keep a runtime-only map to the exact leaf sockets; they never
become fields in `BlockDefinitionV2` or `effectiveGraph`. Reconnect resolves
the handle back to the leaf endpoint, reopening restores the exact original
edge, and execution uses every semantic edge regardless of which containers
are collapsed. Edges whose endpoints are both hidden by the same container are
not projected onto its public surface.

## Phase 1 — Recursive hierarchy compiler and renderer

1. Join each registered workflow with its unpruned pipeline tree and exact
   workflow selection trace.
2. Include active sequential/conditional/auto/loop container definitions in
   the registered `BlockDefinitionV2`; do not expose inactive alternatives as
   active execution nodes.
3. Preserve exact class, kind, contract hash, placement path, parent path, and
   immutable Diffusers revision for every node.
4. Project parents before descendants and assign each child to its immediate
   semantic parent.
5. Add independent expand/collapse controls for every Modular container.
6. Recalculate bounds recursively without allowing collapsed user sizing to
   constrain expanded descendants.
7. Preserve child dragging, resizing, edge routing, z-order, toolbar behavior,
   and root public sockets at every depth.

Acceptance:

- all 94 definitions compile deterministically into 1,794 active graph nodes;
- maximum compiled hierarchy depth is five and no child renders outside its
  immediate visible container;
- Qwen text-to-image visibly renders its active text encoder, denoise branch,
  denoise loop members, decoder branch, loader, and preview at their correct
  levels;
- parameter-only edits modify one value and no graph, interface, or unrelated
  presentation field.
- first expansion is progressively disclosed, and internal Blocks use the
  shared Block frame, controls, connectors, resize behavior, selection, and
  subtree-save action instead of a dashed Group-only renderer.

## Phase 2 — Insertable Modular Diffusers catalog

1. Generate catalog entries from every exact unpruned placed definition.
2. Group entries by pipeline family, class, and block kind while keeping every
   exact context/hash selectable.
3. Compile a leaf drop into an ordinary runtime node with exact typed inputs,
   outputs, configs, and component requirements.
4. Compile a container drop into the shared V2 Block renderer with recursive
   descendants.
5. Permit drops on the top-level canvas and inside compatible expanded
   containers.
6. Keep incompatible/context-incomplete nodes addable. Highlight unresolved
   `PipelineState`, loop state, component, and value connections and expose a
   precise Fix suggestion instead of silently rejecting the drop.

Acceptance:

- every block referenced by all 94 workflows is searchable and draggable;
- a dropped item uses the same renderer and fields as the same item projected
  from a Cluster;
- catalog browsing and dropping never download a model or enable remote code.

## Phase 3 — Composition and semantic validation

Extend the pinned backend composition recipe from `remove`, `move`, and
`duplicate` to exact `insert` and `replace` operations. Validate:

- destination container kind and ordering;
- input/output and `kwargs_type` compatibility;
- `PipelineState` and loop-member ownership;
- component type and immutable artifact requirements;
- prevention of cycles, orphaned children, and invalid cross-container edges.

A valid changed tree is rebuilt through upstream `init_pipeline()`. Invalid
changes remain visible and editable but cannot silently execute the old
registered graph. Studio and Fix must identify the first invalid node/edge and
the concrete compatible action.

## Phase 4 — Save any hierarchy level as a User Node

Add one shared node-toolbar action for Cluster roots, Modular containers,
loops, and leaves:

1. collect the selected node and all semantic descendants;
2. include internal edges and derive boundary ports from crossing edges;
3. rebase placement paths and layouts relative to the selected root;
4. copy current instance values into the new definition defaults, including
   prompts, parameters, reviewed model choice, and included component bindings;
5. retain immutable source revision and parent-definition provenance;
6. create a user-owned, mutable `BlockDefinitionV2` with new identities and
   canonical hashes;
7. leave the current workflow instance unchanged after saving.

Save choices:

- registered source: **Save as new User Node** or **Keep in workflow**;
- user source: **Update existing User Node**, **Save as new User Node**, or
  **Keep in workflow**.

Suggested names include workflow and hierarchy context, for example
`Qwen Image / Text to Image / Denoise`.

## Phase 5 — Persistence and migration

1. Normalize existing flat V2 instances deterministically into recursive
   presentation without changing values, ports, definition references, or
   execution identity.
2. Keep legacy V1 instances behind the existing hash-bound archive/equivalence
   recovery boundary; never guess a registered replacement.
3. Persist hierarchy expansion, relative layouts, subtree edits, current
   values, effective interfaces, and user-definition decisions through the
   existing workflow backend.
4. Make undo/redo atomic for expansion, node adoption, structural edits, and
   subtree saving.
5. Preserve independent instances across workflows and tabs.

## Phase 6 — Verification and rollout

Focused contract tests:

- snapshot counts, exact IDs, roots, placements, and maximum depth;
- hierarchy compilation and parent ordering for all 94 workflows;
- client/backend V2 normalization and hash agreement;
- every catalog block is insertable without model loading;
- compatible and incompatible composition validation;
- subtree boundary derivation and current-value defaults;
- update/new/workflow-only save behavior;
- persistence, restart, migration, undo/redo, and multi-instance isolation.

Browser tests:

- drag a Cluster into an empty graph and observe immediate insertion feedback;
- recursively expand to the deepest available level;
- move and resize children at several levels;
- edit one parameter and verify no collateral change;
- insert, replace, remove, and reconnect a Modular block;
- save root, intermediate container, loop, and leaf User Nodes;
- save, refresh, restart, and compare the serialized instance byte-for-byte;
- collapse/re-expand and verify public inputs/outputs and internal topology;
- run collapsed and expanded Qwen workflows through the frontend and compare
  execution/output fingerprints.

After the Qwen vertical slice passes, enable the same generic compiler for all
94 registered Diffusers workflows. Real model qualification and publication
remain separate evidence gates; structural availability never implies that a
model is installed, licensed, resource-qualified, or publicly promoted.

## Estimated completion

- recursive schema/compiler/renderer and Qwen proof: 2–3 focused days;
- all-block catalog, insertion, composition, and subtree saving: 3–5 focused
  days;
- migration, all-94 structural audit, browser regression, and production
  bundle smoke: 2–3 focused days.

Total functional migration: approximately 7–11 focused engineering days,
excluding downloads, legal acknowledgements, hardware-specific execution
qualification, and showcase approval.

## Current implementation checkpoint

- [x] Pinned all-94 and unpruned hierarchy counts audited.
- [x] React Flow recursive-subflow capability verified.
- [x] Current flattening and non-insertable catalog boundaries identified.
- [x] Recursive hierarchy compiler for every exact admitted Modular route.
- [x] Recursive renderer, relative layouts, per-container collapse, recursive
      bounds, and compact parameterless leaves.
- [x] Unpruned Modular Diffusers catalog: 1,051 contextual placements / 598
      unique placed block definitions are searchable and draggable without a
      model download.
- [x] Exact leaf adoption into a selected nested container with immutable
      catalog provenance and save/refresh rematerialization.
- [x] Insert/replace backend recipe validation and upstream-tree rebuild:
      mutate pinned unpruned `pipeline.blocks`, then select the workflow and
      call `init_pipeline()`.
- [x] Effective V2 graph lowering for parameter-only, leaf insert, leaf
      replace, move, and remove operations, with a visible backend rebuild
      receipt action.
- [x] Save-subtree User Node lifecycle for root, intermediate container, loop,
      and leaf selections.
- [x] Existing flat V2 normalization and V1 recovery compatibility remain
      behind their explicit archive/equivalence boundary.
- [x] All-94 upstream hierarchy audit (maximum upstream depth five).
- [x] All 94 reviewed workflows compile to the shared structural
      `BlockDefinitionV2`: 1,794 active ordinary internal nodes, maximum upstream
      depth five, and no nested Cluster/User node payloads.
- [x] Container catalog drops materialize the complete reviewed subtree as a
      source-neutral V2 fragment. Dropping that fragment into an expanded
      compatible Block flattens and rebases its ordinary descendants; the
      exact immutable source triplet survives Save/refresh.
- [x] Catalog-only reviewed Diffusers Clusters remain structurally insertable
      through the same V2 renderer, while carrying no execution/Auto authority.
- [x] Fresh admitted insertion uses a build-time generated catalog containing
      all 90 current route admissions (82 Diffusers and eight Transformers),
      validated by both runtimes and fetched without hidden field actions,
      optional-runtime activation, Hub access, or model loading.
- [x] The compiled catalog is reproducible through
      `npm run catalog:block-v2:generate`; both main validation gates run the
      independent all-route byte-for-byte staleness check and always remove
      their temporary audit directory.
- [x] Invalid V2 topology is reported as a Block-structure issue, targets the
      implicated projected node when possible, and offers a recoverable Fix
      that restores reviewed structure while preserving compatible additions.
- [x] The complete 10-route Qwen family browser lifecycle proves empty-graph
      insertion, creator defaults, public ports, recursive containment,
      parameter edits, Save, refresh, byte-identical instance restoration,
      and contained re-expansion. Current evidence is isolated under
      `data/review/nested-modular-v2-2026-09-04/`.
- [x] Qwen production-browser resize proof covers collapsed-root resizing,
      expansion that ignores the collapsed constraint, nested-node resizing,
      collapse/re-expansion, and retained internal size.
- [x] Collapsed internal Blocks derive real typed boundary sockets for hidden
      descendant connections. Shrinking a projected Block preserves its
      connector tray and crossing links; reconnect targets the exact semantic
      leaf socket; Save/refresh restores the same surface; collapsed and
      expanded execution retain the complete flat graph.
- [x] Structural editing is proven in the complete browser gate: ordinary
      nodes can be adopted into and moved out of nested Blocks, compatible
      nodes can be replaced, public ports can be configured and reconnected,
      and all changes persist through Save/refresh. The V2 root no longer
      applies the legacy User Block auto-fit rule while a descendant is being
      dragged out.
- [x] Generated V2 definitions are the primary insertion path. The client
      verifies and materializes the backend-served immutable definition before
      graph mutation; the field-action compiler remains only a bounded legacy
      recovery fallback.
- [x] Full backend regression: 2,452 passed, 54 skipped, and 6,879 subtests.
- [x] Full client static/unit/build/bundle gate (`npm run check`).
- [x] Full browser gate: two shared-control tests and all 126 Studio mocked
      tests. This includes recursive containment, catalog drops, nested
      composition, all User Node persistence choices, restart/isolation,
      invalid-topology recovery, and legacy migration/rollback.
- [x] Backend-served production-bundle smoke at `127.0.0.1:8088`: collapsed
      root resize, recursive expansion, every projected Qwen node contained,
      internal model-loader resize, collapse/re-expand, and retained internal
      size.
- [ ] Re-run real Qwen media generation through the current generated-catalog
      build and compare collapsed/expanded execution fingerprints. This is an
      execution/publication qualification item, not a structural migration
      blocker.

## Remaining work outside the structural migration

- Qualify real execution, resources, installed immutable artifacts, and output
  quality independently for each route before enabling executable,
  `liveProof`, publication, or Auto flags.
- Re-run Qwen collapsed/expanded generation through the current production
  bundle and compare execution and output fingerprints.
- Preserve historical V1 instances behind archive/equivalence recovery where
  their exact definition is no longer current; never infer equivalence from a
  display name or model family.
- Promote only definition-specific receipts whose generated media is approved.
