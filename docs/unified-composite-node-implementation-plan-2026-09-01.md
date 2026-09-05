# Unified Composite Node Contract and Implementation Plan — 2026-09-01

## Status and authority

This document is the authoritative product and persistence contract for every
composite node shown on the MoDiff canvas. It supersedes the separate Cluster
renderer and destructive Cluster-to-User-Node conversion described in the
older [visual node system plan](hugging-face-visual-node-system-plan.md) and
[2026-08-27 demo plan](demoable-hugging-face-nodes-plan-2026-08-27.md).

The implementation is not complete until the status checklist at the end of
this document is green. Earlier tests that proved the legacy conversion flow
remain useful historical evidence, but they do not satisfy this contract.

The normative terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** have their
usual requirements meaning.

### Canonical implementation sources

This document defines the product semantics. The executable V2 contract is
maintained in the following synchronized source files:

- `MoDiff-client/src/studio/blockSchemaV2.ts` — TypeScript types, strict
  definition and instance validators, canonical serialization/hashing,
  instance construction, and the additive V1 User Node reader;
- `MoDiff/modiff/block_definition_v2.py` — backend `TypedDict` definitions,
  strict reusable-definition validation, canonical serialization/hashing, and
  the user-store ownership boundary;
- `MoDiff/modiff/server.py` — `/studio/blocks` V1/V2 dispatch and persistence;
- `MoDiff-client/src/studio/registeredBlockAdapterV2.ts` — the model-family-
  neutral compiler from one exact registered Diffusers or Transformers
  admission plus its complete materialized execution skeleton into
  `BlockDefinitionV2` and an independent `BlockInstanceV2`;
- `MoDiff-client/src/studio/registeredBlockV2Routes.ts` — the explicit,
  fail-closed catalog route ledger. Each entry pins one schema-v6 definition
  content hash, admission, immutable library/artifact revisions, exact Studio
  receipt, dynamic-field actions, and reviewed executable public boundary.
  The ledger MUST NOT infer a route from a pipeline class or family name;
- `MoDiff-client/src/studio/registeredBlockV2FanOutRoutes.ts` — the generated
  exact-route extension for admissions that require declared logical fan-out,
  aliased public IDs, or terminal-media adaptation. It remains subject to the
  same route audit, V2 content hash, and canonical SHA-256 pins as the primary
  ledger and is never a permissive family fallback;
- `MoDiff-client/src/studio/blockValueTypeCompatibilityV2.ts` — the V2-only
  canonicalization boundary for Modular Diffusers scalar type aliases. It MUST
  NOT broaden or rewrite the global canvas type system;
- `MoDiff-client/src/studio/huggingFaceClusterInsertion.ts` — the registered
  catalog insertion boundary. Each exact admission must compile atomically to
  one durable V2 `block` root; transient compiler nodes must never persist,
  export, enter undo history, or remain after failure;
- `MoDiff-client/src/studio/huggingFaceNodeCatalog.ts` — the discovery versus
  insertion boundary. Catalog metadata remains searchable, but a row is
  graph-qualified, draggable, and insertable only when one exact admission
  resolves through the pinned V2 route ledger. A catalog-only definition MUST
  remain disabled and MUST NOT create a legacy `cluster` root as a fallback;
- `MoDiff-client/src/studio/compositeBlockCapabilitiesV2.ts` — the
  source-neutral, fail-closed action resolver. Capabilities are transient UI
  policy and MUST NOT be serialized into either V2 contract or an authority
  receipt;
- `MoDiff-client/src/studio/blockDefinitionPersistenceV2.ts` and
  `blockPersistenceV2.ts` — the reusable-definition persistence adapter and
  three-choice workflow coordinator. They preserve the effective graph,
  explicit interface, declared current defaults, preview bindings, and parent
  provenance; reject registered-definition updates; validate the exact server
  response; and rebase only the initiating instance after race checks;
- `MoDiff-client/src/studio/blockRuntimeV2.ts` — the source-neutral V2 canvas
  projection, mutation, execution-expansion, and persistence-canonicalization
  boundary. It derives React Flow roots, controls, connectors, children, and
  bridge edges from one `BlockInstanceV2`; those projections are never a
  second persisted authority;
- `MoDiff-client/src/studio/nodeConnectorResolution.ts` and
  `MoDiff-client/src/stores/flowConnectionMutations.ts` — the common connector
  lookup and mutation boundary. Public V2 sockets are derived from the
  definition's explicit boundary, external connections terminate on the
  durable root, and admitted same-owner internal edge changes update the
  instance effective graph copy-on-write;
- `MoDiff-client/src/components/BlockNode.tsx`, `BlockNodeFrame.tsx`, and
  `BlockNodeV2.tsx` — the single React Flow `block` renderer entry, shared
  visual/action frame, and source-neutral V2 projection view. Source kind may
  gate actions but MUST NOT select another frame;
- `MoDiff-client/src/stores/useFlowStore.ts` — the workflow integration point
  that invokes V2 persistence canonicalization and copy-on-write presentation
  and value actions for already-V2 roots, expands those roots into their
  concrete graph during API export, resolves root execution targets to a
  deterministic preview child, and applies a saved reusable definition to one
  targeted instance without refreshing sibling snapshots;
- `MoDiff-client/src/stores/useUserBlockStore.ts`,
  `src/components/NodeList.tsx`, and `src/workflow/useWorkflowDrop.ts` — the
  mixed V1/V2 User Node library boundary. User-owned V2 definitions are
  fetched, optimistically saved with failure rollback, listed, and reinserted
  by click or drag as new independent top-level `BlockInstanceV2` roots; and
- `MoDiff-client/src/studio/runReadiness.ts` and `runCoordinator.ts` — the
  already-V2 readiness and submission boundary. Both inspect the same concrete
  execution graph used by export, reject malformed V2 authority fail-closed,
  and do not send a composite wrapper to the backend;
- `MoDiff-client/src/studio/graphFixer.ts` — the explicit recovery boundary for
  malformed workflow-local V2 structure. It validates the effective graph
  through execution expansion, targets the owning root, restores registered
  baseline nodes/edges only after user selection, and retains compatible
  custom additions. It MUST NOT mutate `definitionSnapshot`, update a reusable
  definition, or silently repair a graph during Run;
- `MoDiff-client/src/studio/blockAutoAuthorityV2.ts` and
  `MoDiff/modiff/huggingface_cluster_runtime.py` — the instance-scoped Auto
  authority boundary. The client submits the complete normalized instance and
  explicit planner form; the backend validates and recomputes all identity,
  canonical-definition, resource, runtime, admission, and artifact material
  before issuing a short-lived receipt;
- `MoDiff-client/src/studio/blockAutoEligibilityV2.ts` and
  `blockRunFormV2.ts` — the source-neutral registered-instance eligibility and
  exact run-form/provenance projectors. Auto accepts one current pinned root
  with parameter-only customization and reviewed companion nodes; Expert
  remains executable without Auto authority. Output history is repaired from
  the immutable V2 graph snapshot and exact preview-node identity rather than
  unrelated Studio form state;
- `MoDiff-client/src/studio/userBlocks.ts` and
  `src/stores/websocketMessageHandler.ts` — deterministic mapping from concrete
  V2 runtime nodes back to their durable collapsed owner and declared preview
  binding. WebSocket output/progress is stored in instance-local
  `previewStates`, never mirrored into root `NodeData.params` or undo history;
- `MoDiff/modiff/composite_migration_inventory.py` and
  `MoDiff/scripts/inventory_legacy_composites.py` — the deterministic,
  read-only legacy inventory builder and stdout-only operator CLI. They inspect
  saved workflow JSON and reusable User Node records without converting,
  deleting, merging, recovering, or writing either store;
- `MoDiff/modiff/composite_migration.py`,
  `studio_persistence_lock.py`, `workflow_store.py`, and `server.py` — the
  separately authorized V1-to-V2 and exact compiler-supplemented registered-
  Cluster preview/application and exact-backup recovery boundary. It shares
  one process-local lock with ordinary workflow/User Node writes, verifies
  reviewed source/target/compiler hashes, and never force-overwrites, deletes,
  renames, or merges a record; and
- `MoDiff-client/src/studio/registeredClusterCompilerSupplement.ts` and
  `src/components/CompositeMigrationCard.tsx` — the visible exact-current
  legacy compiler-supplement generator and review/recovery UI. Generation
  reads exact saved workflow bytes, uses hidden transient compilation, binds
  every mapping receipt, performs a final backend re-preview, and never writes
  a workflow before the separately confirmed Apply operation; and
- `MoDiff/tests/test_composite_migration_inventory.py` — mixed legacy
  Cluster/V1 User Node/already-V2 inventory fixtures, ambiguity/error coverage,
  deterministic hashing, input immutability, and filesystem/CLI non-mutation
  evidence. `test_composite_migration.py` and
  `test_composite_migration_server.py` cover synchronized V1 conversion,
  registered compiler supplements, identity/value/interface/layout/edge
  preservation, pin/tamper rejection, explicit authority, exact backups,
  stale-plan conflicts, idempotency, rollback, reapply, and the HTTP contract.

Live Auto qualification remains a production gate. The schema-v6 compiler
audit now covers all 90 exact registered definition/admission pairs: 82
Diffusers and eight Transformers. This includes Qwen Image and Flux,
MiniMax Music 3 text-to-audio, and all three currently admitted speech routes.
Each admission remains an explicit independently reviewed route; broad routing
does not permit model-family inference, partial dynamic-field finalization, or
reuse of another admission's compiler pins.

Catalog visibility is not execution authority. Definitions without an exact
current `registeredBlockV2Route` remain visible as **Catalog only** and are
structurally draggable/insertable through the shared source-neutral V2
renderer, but receive no Run, Auto, or publication authority. Programmatic
execution and promotion boundaries repeat the exact-route check.
This defense-in-depth rule prevents a stale hash, missing route, or future
discovery-only definition from reviving the legacy renderer and reintroducing
different controls, ports, persistence, or expansion behavior.

The client and backend definition canonicalizers MUST produce identical
`graphHash` and `contentHash` values for the same JSON payload. Neither
implementation is a fallback copy that may drift. Any reusable-definition
schema or hashing change MUST update both sides and their cross-runtime
fixtures in the same change. Backend migration also uses the same canonical
JSON/FNV contract for `blockInterfaceHashV2`; newly migrated durable instances
MUST materialize `effectiveInterface` rather than relying on the client's
missing-field legacy normalizer. The registered adapter is also part of that
synchronized change whenever a source, graph, boundary, control, suggestion,
preview, or provenance field changes.

`/studio/blocks` stores reusable user-owned definitions, not workflow
instances. `BlockInstanceV2` is embedded in the saved workflow so its values,
layout, preview state, and effective graph remain insertion-local. Registered
definitions use the same schema in memory but remain read-only catalog data.

### Historical legacy mismatch and integration boundary

The 2026-09-01 pre-V2 audit found two independent legacy systems whose
differences explain the original defect. New registered insertions no longer
use this split; the description remains normative migration context for saved
legacy records:

- a registered Cluster inserts as React Flow `type: "cluster"`, renders through
  `HuggingFaceClusterNode`, stores `huggingFaceClusterInstance`, reconstructs
  derived execution children from the live catalog, and translates its
  boundary during export; and
- a User Node inserts as `type: "block"`, renders through `BlockNode`, stores a
  V1 `userBlockSnapshot`, and expands/collapses through the V1 User Block
  functions, which may normalize or re-derive its public interface.

This is why moving from the former representation to the latter can change
controls, handles, action placement, dimensions, and internal fields. It is
not an acceptable source-category difference and it MUST NOT be preserved in
the V2 implementation.

During the migration window, the legacy registered path has a containment fix:
concurrent capability discovery shares one in-flight request, workflow-epoch
changes rematerialize every registered instance independently, and moved
execution-child positions persist as parent-relative presentation state. The
mocked two-instance Save/refresh test proves that this prevents the earlier
cross-instance rematerialization/reset race. It does not make the legacy
`cluster` root or `HuggingFaceClusterNode` the accepted V2 implementation.

Changing only the legacy Cluster root's `type` to `block` is also invalid.
`BlockNode` currently assumes a V1 `userBlockSnapshot`; doing that would create
two competing authorities (`BlockInstanceV2` versus V1 snapshot/params) whose
values could diverge on the first edit, collapse, refresh, duplicate, or run.
The migration gate therefore requires the V2 runtime projector and controller
before any registered insertion is switched to the common renderer.

For a canonical V2 root, the only persisted composite payload is:

```ts
{
  type: "block",
  data: {
    type: "block",
    blockInstanceV2: BlockInstanceV2,
  },
}
```

Catalog labels, badges, controls, connectors, previews, child nodes, and bridge
edges are derived views. A V2 root MUST NOT also contain
`huggingFaceClusterInstance`, `userBlockSnapshot`, or another mutable copy of
its values or graph.

## Product decision

MoDiff has one composite-node system. A registered Diffusers Cluster, a
registered Transformers Cluster, an imported Hub block, and a User Node use
the same canvas type, renderer, interface model, expansion mechanics,
persistence format, and connection behavior.

`Cluster` is a catalog and provenance classification. It is not a second kind
of canvas widget.

The product differences are limited to:

- where the reusable definition is registered;
- who owns and may update the reusable definition;
- immutable source and artifact provenance;
- reviewed execution and Auto authority; and
- catalog badges, discovery categories, and permitted save actions.

Those differences MUST NOT select a different renderer, infer a different
public interface, alter saved parameter values, or replace a canvas node.

## User-visible invariants

1. Two insertions of the same definition initially have the same controls,
   ports, actions, minimum size, resize behavior, and expansion affordance.
2. Running one instance or viewing its output may update its run status and
   preview only. It MUST NOT alter its definition, public interface, or
   presentation contract.
3. Expanding a block reveals ordinary connected graph nodes with their normal
   controls and sockets. Collapse is presentation-only.
4. Moving an existing internal node changes only that instance's internal
   layout. It MUST NOT create a User Node, call `/studio/blocks`, recalculate
   controls or ports, or invalidate execution authority.
5. Editing an exposed value changes only that workflow instance. The saved
   value wins over definition defaults after Save and browser refresh.
6. Adding, removing, replacing, or reconnecting an internal node is a
   copy-on-write structural edit of the same workflow block instance. The
   reusable registered source remains immutable.
7. A structural edit may invalidate reviewed execution or Auto authority, but
   MUST NOT change the renderer, canvas type, public boundary, name, prompt,
   dimensions, or external connections.
8. A reusable User Node is created or updated only after an explicit save
   choice. Structural editing alone MUST NOT add a library entry.
9. Inputs and outputs change only through the explicit **Configure interface**
   action. Internal movement or topology edits never infer a new boundary.
10. Cluster Nodes and User Nodes cannot be nested. Ordinary nodes can be moved
    into or out of an expanded composite with one undoable mutation.

## Terminology

| Term                | Meaning                                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Block definition    | Reusable graph, explicit interface, exposed controls, defaults, preview bindings, and source metadata.              |
| Block instance      | One canvas insertion with its own values, effective graph snapshot, layout, size, preview, and customization state. |
| Registered Cluster  | A block definition owned by the reviewed Diffusers or Transformers catalog.                                         |
| User Node           | A reusable block definition owned by the user.                                                                      |
| Workflow-only block | A customized instance whose reusable definition has not been created or updated.                                    |
| Structural edit     | Addition, deletion, replacement, adoption, or reconnection that changes the effective internal graph.               |
| Presentation edit   | Move, resize, expand, collapse, pan, selection, or viewport change.                                                 |
| Public boundary     | Ordered, stable input and output ports visible outside the composite.                                               |
| Authority           | A separately calculated permission or receipt for reviewed execution, Auto management, or publication.              |

## Canonical persisted contracts

The following TypeScript-shaped definitions are normative at the semantic
level. Implementation types may split them into files, but field meaning and
invariants MUST remain equivalent. Persisted JSON uses `schemaVersion: 2`.

```ts
type BlockJsonPrimitive = string | number | boolean | null;
type BlockJsonValue =
  BlockJsonPrimitive | BlockJsonValue[] | { [key: string]: BlockJsonValue };
type BlockJsonObject = { [key: string]: BlockJsonValue };
```

Definition and instance payloads accept JSON values only. Runtime objects,
callbacks, class instances, tensors, and non-finite numbers are rejected.

### `BlockDefinitionV2`

```ts
type BlockDefinitionV2 = {
  schemaVersion: 2;
  definitionId: string;
  displayName: string;
  description?: string;
  contentHash: string;

  source: BlockSourceV2;
  graph: BlockGraphV2;
  boundary: BlockBoundaryV2;
  controls: BlockControlV2[];
  suggestedInputs?: SuggestedInputSetV2[];
  previews: BlockPreviewBindingV2[];

  // Describes reusable-definition ownership, not canvas rendering.
  ownership: {
    kind: "registered" | "user";
    definitionMutable: boolean;
  };
};
```

Required invariants:

- `definitionId` is stable and opaque. Display names are never identity.
- `contentHash` covers exactly the canonical graph (without its derived
  `graphHash` field), boundary, ordered controls including declared defaults,
  ordered previews, and these execution-relevant source bindings when present:
  `provider`, `library`, `libraryRevision`, `pipelineClass`, `blocksClass`,
  `workflow`, `manifestDefinitionId`, `manifestContentHash`, `repository`, and
  `repositoryRevision`, plus `executionAdmissionId` when declared.
- `contentHash` deliberately excludes `schemaVersion`, `definitionId`,
  `displayName`, `description`, `suggestedInputs`, `ownership`, and the source's
  `kind`, `catalogCategory`, and `parent` ancestry. Those fields remain strictly
  validated and persisted even though they are not execution-content identity.
- A registered definition has `ownership.kind = "registered"` and
  `definitionMutable = false`.
- Every `user` or `hub_import` source has `ownership.kind = "user"` and
  `definitionMutable = true`. Updating it is still an explicit user operation;
  `definitionMutable` does not authorize silent writes.
- The graph contains stable semantic node and edge IDs. IDs do not depend on
  canvas position, expansion state, workflow title, or insertion order.
- Defaults exist only in the definition. Current workflow values exist only in
  the instance.
- The definition does not contain a volatile resource plan, loaded component,
  model object, tensor, task status, or live execution receipt.

### `BlockSourceV2`

```ts
type BlockSourceKindV2 =
  "diffusers_catalog" | "transformers_catalog" | "hub_import" | "user";

type BlockSourceV2 = {
  kind: BlockSourceKindV2;
  catalogCategory?: "diffusers" | "transformers";
  provider?: string;
  library?: "diffusers" | "transformers";
  libraryRevision?: string;
  pipelineClass?: string;
  blocksClass?: string;
  workflow?: string;
  manifestDefinitionId?: string;
  manifestContentHash?: string;
  // Exact reviewed execution admission. Required for registered definitions
  // that may receive reviewed-execution or Auto authority.
  executionAdmissionId?: string;
  repository?: string;
  repositoryRevision?: string;
  parent?: {
    definitionId: string;
    contentHash: string;
    sourceKind: BlockSourceKindV2;
  };
};
```

Source invariants:

- Repository-backed execution references use immutable revisions. A moving
  branch is not a reusable execution identity.
- `repository` and `repositoryRevision` are an all-or-nothing pair;
  `hub_import` requires the pair, and `repositoryRevision` is an exact
  lowercase 40-hex commit.
- Registered sources require matching catalog/library kinds plus
  `libraryRevision`, `manifestDefinitionId`, `manifestContentHash`,
  `pipelineClass`, and `workflow`. Hub and User sources cannot claim a
  registered `catalogCategory`.
- `executionAdmissionId`, when present, is an exact immutable catalog identity,
  is covered by `contentHash`, and must match the admission used to compile the
  registered graph. It is not inferred from `definitionId`, display text, model
  class, or workflow mode. Reviewed-execution and Auto receipts for a registered
  definition fail closed when it is absent or stale.
- `parent` records ancestry when a reusable User Node is saved from a
  registered or imported definition. It does not make the child definition
  registered or reviewed.
- Source metadata is never removed merely because an instance is customized.
- Source kind controls catalog placement and provenance display only. It MUST
  NOT select a canvas renderer or port-inference algorithm.

### `BlockGraphV2`

```ts
type BlockGraphV2 = {
  nodes: BlockGraphNodeV2[];
  edges: BlockGraphEdgeV2[];
  executionOrder?: string[];
  graphHash: string;
};

type BlockGraphNodeV2 = {
  nodeId: string;
  nodeType: string;
  data: BlockJsonObject;
  semanticRole?: string;
  upstreamBlockPath?: string;
  modularDiffusers?: BlockGraphNodeModularDiffusersV2;
};

type BlockGraphEdgeV2 = {
  edgeId: string;
  sourceNodeId: string;
  sourcePortId: string;
  targetNodeId: string;
  targetPortId: string;
};
```

`modularDiffusers`, when present, is canonical, hash-covered semantic identity
for one expanded registered/imported Modular Diffusers node. It distinguishes
MoDiff infrastructure such as component loading and preview from an exact
upstream block. Every entry pins the pipeline class, blocks class, workflow,
Diffusers revision, and runtime role. An upstream entry additionally pins the
block definition ID, concrete class, kind, contract hash, exact placement path,
parent placement path, and component names. It is provenance and lowering
metadata, never a parallel value or layout store. Registered semantic graphs
must not replace exact upstream block placements with broader convenience
stages; any optimized runtime lowering is explicit, versioned, hash-bound, and
must preserve the meaning of every supported edit.

Graph positions and reusable layout hints do not belong in
`BlockDefinitionV2`. Initial layout is supplied while constructing an instance
and is persisted only in `BlockInstanceV2.presentation`; instance layout always
wins.

Recursive Modular UI state is likewise instance presentation, not definition
identity. `collapsedContainerNodeIds` contains semantic graph-node IDs whose
exact `placementPath` is referenced by at least one descendant's
`parentPlacementPath`. The owning node may be a structural `group` or an
executable `custom` block; canvas renderer type is not container authority.
New hierarchical instances seed all such owners as collapsed for progressive
one-level expansion. Client and backend validators must derive and validate
that same set from the hash-covered graph metadata.

`graphHash` covers node and edge content plus the optional ordered
`executionOrder`, but not the `graphHash` field itself. Canonicalization sorts
the node array by `nodeId` and the edge array by `edgeId`, so their storage order
is not identity; explicit `executionOrder` order remains semantic. Node IDs and
edge IDs are independently unique, edges reference declared nodes, and an
explicit execution order contains unique declared node IDs. Nested composite
nodes are rejected.

### `BlockBoundaryV2`

```ts
type BlockBoundaryV2 = {
  mode: "explicit" | "derived";
  inputs: BlockPortV2[];
  outputs: BlockPortV2[];
  derivation?: {
    algorithmVersion: string;
    derivedAtDefinitionHash: string;
  };
};

type BlockPortV2 = {
  portId: string;
  label: string;
  valueType: string;
  required: boolean;
  multiple?: boolean;
  binding: {
    nodeId: string;
    fieldOrPortId: string;
  };
  // Inputs only. The one public logical value is projected atomically to the
  // primary binding above and every canonically ordered mirror below.
  mirrorBindings?: {
    nodeId: string;
    fieldOrPortId: string;
  }[];
};
```

Boundary rules:

- Registered Diffusers and Transformers definitions MUST use `explicit`.
- Hub imports that declare a validated interface SHOULD use `explicit`.
- A User Node created from an arbitrary graph selection MAY begin as
  `derived`. Derivation runs once while creating the reusable definition and
  persists the resulting ordered ports.
- `derived` does not mean continuously recomputed. Graph edits never
  automatically add, remove, or reorder ports.
- Re-derivation is an explicit **Configure interface** operation, produces a
  preview of affected external edges, and is one undoable mutation.
- `portId` remains stable when its label changes. Deleting a connected port
  requires explicit confirmation and a useful connection-loss report.
- Every public `portId` is unique across both `inputs` and `outputs`, not only
  within one direction. The shared root connector map and React Flow handle
  namespace cannot represent an input and output with the same handle ID
  without ambiguity, so both client and backend validators reject that
  definition.
- `mirrorBindings` is permitted on public inputs only. It represents genuine
  semantic fan-out of one logical input to multiple internal consumers; it is
  not a fallback list and the primary binding remains required. Mirrors are
  non-empty when present, sorted lexically by `nodeId` then `fieldOrPortId`,
  unique, distinct from the primary, bound to declared non-output fields, and
  type-compatible with the public port. Public outputs remain single-binding
  until an explicit aggregation contract is designed and reviewed.
- The collapsed socket still has exactly one `portId`. An external edge into
  that socket, a stored instance value, or an internal adoption gesture fans
  out atomically to the primary and every mirror. Expanded root bridge edges
  show all of those targets. Execution export expands the one external input
  into deterministic internal edges; move-out and collapse coalesce the exact
  fan-out back to the one public connection. Partial or divergent fan-out
  fails closed instead of silently selecting one consumer.
- Copy-on-write customization of a registered instance starts with the same
  exact boundary. It MUST NOT expose every unconsumed internal input or every
  internal output.

### `BlockControlV2`

```ts
type BlockControlV2 = {
  controlId: string;
  label: string;
  binding: {
    nodeId: string;
    fieldId: string;
  };
  // Additional internal fields that consume the same logical control value.
  mirrorBindings?: {
    nodeId: string;
    fieldId: string;
  }[];
  valueType: string;
  defaultValue?: BlockJsonValue;
  required?: boolean;
  sealed?: boolean;
  order: number;
  group?: string;
  help?: string;
};

type SuggestedInputSetV2 = {
  suggestionId: string;
  label: string;
  source?: string;
  values: Record<string, BlockJsonValue>; // keyed by controlId
};

type BlockPreviewBindingV2 = {
  nodeId: string;
  outputPortId: string;
  mediaType: "image" | "video" | "audio" | "text" | "file";
  primary?: boolean;
};
```

Control rules:

- The collapsed surface and Studio inspector consume the same ordered control
  descriptors and instance values.
- A control with `mirrorBindings` is still one logical control and has one
  `controlId`, one displayed value, and one entry in `BlockInstanceV2.values`.
  Runtime projection writes that value atomically to its primary field and all
  mirrors. At compile time, actual/default values across the declared targets
  must agree; disagreement is an invalid registered route rather than a reason
  to pick one field.
- Control mirrors are non-empty when present, canonically sorted by `nodeId`
  then `fieldId`, unique, distinct from the primary, bound to declared
  non-output fields, and type-compatible with the control. Registered compiler
  routes declare the complete target set explicitly; the adapter never infers
  fan-out merely because labels or current values happen to match.
- `defaultValue` is a starter value, not a load-time reset policy.
- A missing instance value may resolve to the definition default only when the
  field was never set. A saved explicit empty string, `false`, `0`, or `null`
  MUST NOT be treated as missing.
- Suggested creator prompts belong in `suggestedInputs`; applying one is an
  explicit user action that writes instance values. `source` records where the
  suggestion came from; it does not grant execution authority.
- Control IDs are unique and `order` is contiguous from zero. Suggested values
  may reference declared control IDs only, not boundary-only input IDs.
- A control may share an ID with a public input only when both bind the same
  internal node and field; validators reject a shared ID with divergent
  bindings.
- Sealed controls may be displayed for provenance or runtime setup but cannot
  be silently unlocked by customization. Changing a sealed artifact or class
  requires an explicit supported workflow/model replacement path.

### `BlockInstanceV2`

```ts
type BlockInstanceV2 = {
  schemaVersion: 2;
  instanceId: string;
  definitionRef: {
    definitionId: string;
    contentHash: string;
  };

  // A self-contained insertion; prevents library drift or deletion from
  // resetting an existing workflow.
  definitionSnapshot: BlockDefinitionV2;
  effectiveGraph: BlockGraphV2;
  effectiveInterface: {
    boundary: BlockBoundaryV2;
    controls: BlockControlV2[];
    baseInterfaceHash: string;
    effectiveInterfaceHash: string;
  };
  values: Record<string, BlockJsonValue>;

  customization: {
    state: "unchanged" | "parameters_changed" | "structure_changed";
    baseGraphHash: string;
    effectiveGraphHash: string;
  };

  presentation: {
    expanded: boolean;
    position: { x: number; y: number };
    // Durable collapsed/user-resized size. This is not the expanded wrapper
    // size and MUST NOT be overwritten by expansion or child measurement.
    size: { width: number; height: number };
    internalLayout: Record<
      string,
      { x: number; y: number; width?: number; height?: number }
    >;
  };

  previewStates: BlockPreviewStateV2[];
  authorities: BlockAuthorityReceiptV2[];
  routeSelection?: {
    schemaVersion: 1;
    routeSetId: string;
    selectedRouteKey: string;
    inactiveDrafts: Record<string, BlockRouteDraftV1>;
  };
};

type BlockRouteDraftV1 = {
  schemaVersion: 1;
  routeKey: string;
  definitionRef: BlockInstanceV2["definitionRef"];
  definitionSnapshot: BlockDefinitionV2;
  effectiveGraph: BlockGraphV2;
  effectiveInterface: BlockInstanceV2["effectiveInterface"];
  values: Record<string, BlockJsonValue>;
  customization: BlockInstanceV2["customization"];
  internalLayout: BlockInstanceV2["presentation"]["internalLayout"];
};

type BlockPreviewStateV2 = {
  binding: BlockPreviewBindingV2;
  mediaReference?: string;
  taskId?: string;
  status?: "idle" | "queued" | "running" | "complete" | "failed";
};
```

Instance invariants:

- Every insertion receives a distinct `instanceId` and independent `values`,
  `effectiveGraph`, `presentation`, preview, and authority state.
- `definitionSnapshot` makes workflow reload deterministic. Refresh never
  fetches new defaults over saved instance values.
- `effectiveGraph` initially equals the definition graph. A structural edit
  replaces it copy-on-write inside that instance; it does not change the
  reusable definition or canvas node type.
- `effectiveInterface` initially equals the definition boundary and controls.
  Configure Interface replaces it copy-on-write, preserves stable port/control
  IDs unless the user explicitly adds or removes an entry, and never mutates
  `definitionSnapshot`. Its base hash is always the definition interface hash;
  its effective hash covers the ordered boundary and ordered controls.
- Presentation edits update `internalLayout` only and do not change
  `effectiveGraphHash`, `contentHash`, or authority.
- Parameter edits update `values`. Only execution-relevant values affect an
  execution fingerprint. That fingerprint covers the effective-interface hash
  as well as current values, so a boundary/control customization cannot reuse a
  receipt issued for a different public contract; neither controls nor the
  public boundary are re-derived.
- `values` is keyed by a stable declared logical ID: a `controlId`, a public
  input `portId`, or one ID intentionally shared by both projections. Labels
  and array positions are never value identity. If two logical IDs bind the
  same internal field, their values must agree or execution fails closed.
- Primary and mirror bindings do not add value IDs. Reading, editing, saving,
  loading, and execution use the one public/control logical ID and project it
  to the complete declared target set. A graph that contains only part of a
  declared fan-out, or whose mirrored internal values disagree, is invalid.
- Preview state is instance-local and non-authoritative. Viewing or replacing
  it cannot modify the graph or interface. Multiple preview bindings remain
  ordered by the definition; at most one is the primary collapsed preview.
- Replacing an internal node that owns a preview or sealed control preserves
  that node's stable semantic `nodeId`. The replacement must provide every
  referenced field with compatible type and direction; a preview replacement
  must additionally preserve its declared media and display role. This swaps
  the implementation behind the existing immutable binding instead of
  rebinding the preview or sealed control. A replaced preview's stale media,
  task, and status are cleared to an idle state.
- Runtime output is accepted into preview state only when its deterministic
  projected node ID and output port match one declared preview binding. The
  matching artifact URL is preferred over a bounded value fallback. A
  successful update replaces stale media/task state for that instance only and
  is not an undoable graph edit.
- Authority receipts are keyed by `kind`; an instance may independently carry
  reviewed-execution, Auto, and publication receipts. A receipt never grants a
  different renderer or public interface.
- External workflow edges target `instanceId + portId`; they do not target a
  renderer-specific node identity.
- `routeSelection` is an optional instance-only generic shell. Its selected
  route is always the instance's active exact definition; an inactive draft is
  a bounded, non-recursive registered route snapshot and never a nested Block.
  Drafts exclude instance IDs, canvas position/size, previews, task IDs, and
  authority receipts. Switching preserves the one root ID and compatible
  external edges, clears volatile output/authority state, and restores each
  route's values, effective graph/interface, and internal layout independently.
- Same-pipeline/same-workflow checkpoint selection does not use
  `routeSelection`. It is one ordinary bound `modelVariant` control whose exact
  repository options are part of the canonical `BlockDefinitionV2` graph and
  whose chosen value lives in `BlockInstanceV2.values`. Changing it must leave
  the definition reference, effective graph/interface, all other values, and
  presentation byte-for-byte unchanged. Execution independently resolves the
  selected repository to an immutable reviewed revision.

### Configure-interface instance contract

`BlockInstanceV2.effectiveInterface` is the explicit, strictly validated
instance-only interface snapshot. The value namespace, renderer controls,
public connectors, external-edge validation, persistence, and execution
translation all consume this one snapshot. Older V2 workflow instances that
do not contain the field migrate deterministically to the embedded definition
boundary and controls; an unknown field, stale base/effective hash, missing
binding, incompatible field type, duplicate ID/order, or divergent shared
input/control binding fails closed.

Configure Interface edits the ordered boundary and controls copy-on-write.
Removing a currently connected port is blocked with an edge-impact count until
the user disconnects it. Sealed controls cannot be removed or rebound. The
reusable `BlockDefinitionV2` remains unchanged until the user explicitly
chooses **Save as new User Node** or, for an already user-owned definition,
**Update existing User Node**.

The editor exposes entry reordering and the complete multi-target contract.
For a public input or editable control, the user can add or remove additional
type-compatible internal consumers while retaining one logical socket/control
and one instance value. The editor excludes the primary and existing mirrors,
stores additions in canonical binding order, never offers mirrors for public
outputs, and disables mirror mutation for sealed controls.

Re-labeling or reordering an interface entry preserves its declared mirrors.
Rebinding the primary explicitly clears its mirrors until the user reviews and
declares the new complete fan-out. Removing/rebinding an entry, replacing or
deleting an internal node, and sealed-control checks examine the primary and
every mirror together. Interface hashes cover all mirror bindings, so a fan-out
change invalidates stale execution authority even when the public ID and label
do not change.

This is an instance-only contract change. It follows the synchronized
instance-change rules below and requires workflow migration/lifecycle tests,
but it MUST NOT add a mutable interface field to the registered definition or
silently reuse `NodeData.params` as interface authority.

### `BlockAuthorityReceiptV2`

```ts
type BlockAuthorityReceiptV2 = {
  kind: "reviewed_execution" | "auto" | "publication";
  definitionId: string;
  definitionContentHash: string;
  effectiveGraphHash: string;
  executionParameterHash: string;
  artifactRevisions: Record<string, string>;
  admissionId: string;
  issuedAt: string;
  expiresAt?: string;
};
```

Authority rules:

- Provenance says where a block came from. Authority says what its exact
  current graph and parameters are permitted to do. They are not equivalent.
- A parameter change invalidates only receipts whose parameter hash no longer
  matches.
- A structural edit invalidates reviewed execution, Auto, and publication
  receipts unless the effective graph independently matches another reviewed
  admission.
- Invalid authority may block Auto or a reviewed route. In non-Auto/Expert
  mode, the ordinary graph remains runnable when its concrete dependencies are
  available, and real preparation/runtime errors must be shown to the user.
- A finding deliberately demoted to a non-blocking Expert warning remains
  visible and actionable in Graph Fix. Model, environment, and media warnings
  retain their explicit **Models**, **Setup**, or **Gallery** action, but do not
  disable **Run** and are never applied silently. A malformed concrete V2 graph
  remains a blocking structural error; warning presentation does not weaken
  strict graph validation.
- The issuing backend MUST strictly validate the submitted V2 execution
  material and recompute `definitionContentHash`, `effectiveGraphHash`, and
  `executionParameterHash`; it cannot treat client-provided hash strings as
  evidence and echo them into a receipt. It must also resolve the exact
  catalog admission, immutable artifacts, installed revisions, execution
  profile, optional runtime, and selected resource candidate. Hash or graph,
  interface, value, artifact, recipe, or admission substitution fails closed.
- Registered Auto admission pins both the public V2 definition content hash
  and a SHA-256 of the complete canonical definition material. The submitted
  planner form includes every resource-impacting field explicitly; a missing
  dimension, step/frame count, dtype, device, quantization, or offload field is
  an invalid authority request rather than permission to assume a default.
- Multiple independently scoped receipts may coexist in `authorities`. Losing
  or replacing one receipt does not silently remove another kind.
- Losing authority never changes the block renderer, boundary, controls,
  values, source provenance, or external connections.

## One renderer and one canvas type

All composite definitions MUST insert as the existing canonical `block` canvas
type and render through one shared block component. Catalog badges, immutable
source details, qualification status, and save permissions are injected as
capabilities.

The shared surface includes:

- header, source badge, name, status, and action placement;
- selection toolbar and keyboard actions;
- width and height resize handles;
- collapsed controls and suggested inputs;
- input and output connector trays;
- generated preview/output presentation;
- expansion into ordinary connected internal nodes;
- internal node movement and resizing;
- error, progress, and retry presentation; and
- accessible labels and focus order.

The durable root connector trays remain mounted in both collapsed and expanded
states. External workflow edges always terminate on the root's stable public
port IDs; expansion only adds ordinary internal projections and bridge links.
Hiding the root handles while expanded is invalid because it makes existing
external edges point at missing handles and prevents composing the expanded
Block with the surrounding workflow.

The same connector contract applies recursively. A collapsed internal Block
projects only connections that cross its subtree boundary, using typed visible
handles with runtime-only bindings to the exact hidden leaf sockets. It must
not discard those links, expose links wholly internal to the hidden subtree,
or rewrite semantic endpoints to the container. Resize minimums include the
header and connector tray. Presentation collapse and sizing never filter the
flat graph used by execution.

The expanded root frame MUST contain the rendered bounding box of every owned
ordinary child. Its canvas width and height are a derived projection of the
durable `internalLayout` plus measured child dimensions and safe header,
connector, and edge padding. `presentation.size` remains the collapsed,
user-resized size; expanding, measuring, or moving one child MUST NOT rewrite
that size, values, effective interface, definition identity, preview state, or
any sibling instance. Collapse MUST restore the exact prior collapsed size.
Any browser lifecycle that claims expansion support MUST assert containment
before and after a child move and again after workflow Save plus browser
refresh; execution/export parity alone is insufficient.

There MUST NOT be a catalog-specific renderer whose private schema produces a
different visual or graph interface.

A control and a public input may intentionally share the same logical ID and
internal binding (for example, Qwen `prompt`). The shared renderer MUST derive
two views from the one V2 contract: editable body controls and a connector
tray. It MUST NOT force both through one `NodeData.params` entry, because a
single entry cannot simultaneously behave as a textarea and an input socket.

The renderer receives capabilities as a computed view, not as another node
schema:

```ts
type CompositeNodeCapabilitiesV2 = {
  editInstanceValues: boolean;
  editInstanceStructure: boolean;
  configureInterface: boolean;
  keepWorkflowOnly: boolean;
  saveAsNewUserNode: boolean;
  updateReusableDefinition: boolean;
};
```

Capabilities are calculated from definition ownership, instance state, and the
active workspace/user permission boundary. They are not persisted as execution
authority and do not affect canonical graph hashes. A registered definition
normally permits instance edits, workflow-only changes, and **Save as new User
Node**, but never **Update reusable definition**. Registered and user-owned
instances now permit structural editing and Configure Interface when the
workflow/workspace permission boundary allows them; ownership changes only
the reusable-definition save choices. The common connection path derives
public sockets from V2 and commits admitted same-owner internal edge
additions/removals/reconnections copy-on-write. The
safe internal-node deletion path also updates `effectiveGraph` copy-on-write,
clears affected authority, preserves external edges, participates in undo, and
atomically rejects deletion of nodes still referenced by the explicit public
interface, exposed controls, or previews. The pure source-neutral
`addBlockEffectiveGraphNodeV2` reducer can also adopt one persistence-filtered
ordinary semantic node copy-on-write: it validates stable and projection-safe
identity, rejects nested composites, writes explicit execution order and
layout, recomputes graph/customization hashes, and clears authority atomically.
Production gestures route through those reducers: dropping a disconnected
ordinary node adopts it; an edge already connected directly to a declared
root port is translated into an internal edge without inferring a new port;
dropping a compatible ordinary node on an internal projection replaces that
semantic node while preserving compatible edge/public IDs; dragging an
eligible internal node out translates incident links through already-declared
public ports; and React Flow edge reconnection updates either the instance
effective graph or an external workflow edge atomically. Arbitrary external
crossings do not invent an interface: the user disconnects, moves, configures
the explicit interface, and reconnects through the root. Preview-owning and
sealed-control-owning replacements use the stricter identity-preserving path:
the protected semantic ID and binding stay unchanged, all edge/public/control
fields must remain compatible, preview media/display compatibility is checked,
and stale preview run state is cleared. All gestures preserve the wrapper,
participate in undo/redo, invalidate authority on semantic changes, and reject
invalid/nested/incompatible states without a partial mutation.

### Atomic registered-definition compilation gate

A registered block may be inserted through V2 only after its exact execution
node schemas and bindings are complete. For example, the real Qwen text-to-
image admission requires dynamic actions for model type and component signals;
an unfinalized static skeleton is not a valid `BlockDefinitionV2` graph.

The insertion factory MUST resolve the immutable admission/spec/registry,
materialize and finalize an ephemeral persistence-filtered draft, reconcile all
fields and bindings, compile the registered V2 definition/instance, discard the
draft, and then atomically insert one `type: "block"` root. On failure it removes
the draft and shows the exact error. It MUST NOT insert a visible legacy
Cluster first, replace it later, or silently fall back to the legacy renderer.
Legacy `huggingFaceCluster*` field options and projection/receipt markers are
compiler inputs only. The compiler MUST translate their meaning into explicit
V2 controls, boundary, values, provenance, and authority inputs, then omit all
of those legacy-prefixed fields from the source-neutral definition graph.

Initial routing is gated per exact admission (Qwen text-to-image first). A
pipeline-class-wide switch is not permitted until every route with that class
has completed the same dynamic-finalization proof.

## Copy-on-write customization flow

### Parameter or presentation edit

1. Update the instance value or layout.
2. Preserve the registered definition reference and snapshot.
3. Recalculate only affected execution authority.
4. Save the workflow instance normally.
5. Do not create or update a reusable User Node.

### Structural edit

1. Detect an actual graph mutation, not a child drag within the same owner.
2. Clone `effectiveGraph` inside the same instance/history transaction.
3. Apply the mutation and validate typed ports, component/state flow, and
   container semantics.
4. Mark `customization.state = "structure_changed"`.
5. Preserve the explicit boundary, controls, values, provenance, layout,
   preview, external edges, name, and `instanceId`.
6. Clear incompatible authority receipts and show the resulting Expert/Auto
   status without changing presentation.
7. Persist the workflow-only customized instance.

No destructive Cluster-to-User replacement occurs.

### Reusable save choices

After customization the user may choose:

1. **Keep changes only in this workflow** — persist only the embedded
   `BlockInstanceV2`; perform no reusable-definition write.
2. **Save as new User Node** — create a new `BlockDefinitionV2` from the exact
   effective graph and existing explicit interface, retain preview bindings
   and direct source ancestry, apply current instance values as reusable
   defaults for declared controls, assign a new opaque definition ID, and
   point only this instance at an embedded snapshot of that definition.
3. **Update existing User Node** — available only when the current reusable
   definition has a user-owned source and mutable user ownership; update that
   definition explicitly. Other open or saved instances keep their embedded
   snapshots until the user explicitly refreshes them.

A registered definition never offers **Update existing**.

`BlockPortV2` has no reusable `defaultValue`. Therefore a save-as-new or
update-existing operation MUST fail closed when `instance.values` contains a
public boundary input that is not also a declared control. The UI may instead
keep that value workflow-only, or a later explicit Configure-interface flow
may expose it as a control; the persistence adapter MUST NOT silently drop it.

The client writes a reusable definition optimistically but rolls the library
state back when the API fails. After a successful write, it verifies the exact
normalized definition response and a semantic signature of the initiating
instance before rebasing. A concurrent semantic edit aborts the rebase. The
target instance retains its ID, values, presentation, and matching preview
state; sibling insertions and snapshots in other open workflows are not
refreshed. A saved V2 User Node can then be clicked or dragged from the Node
Library to create a new independent top-level instance. Focused store/runtime
tests cover rollback, races, sibling/open-workflow isolation, and independent
reinsertion. A mocked browser lifecycle also exercises all three visible save
choices, adoption into the edited User Node, workflow Save/refresh, and the
rebased instance after **Update existing**. The complete release browser suite
remains a separate gate; that does not make this persistence lifecycle pending.

## Migration plan

Migration is additive, deterministic, idempotent, and non-destructive.

### Inventory and backup

The first non-mutating inventory slice is implemented. Run it from the backend
repository with:

```bash
./scripts/with-runtime-env.sh ./.venv/bin/python \
  scripts/inventory_legacy_composites.py --data-dir /path/to/data
```

The CLI reads only `user-workflows/*.json` and `studio/blocks/*.json` below the
selected data directory and writes the report to stdout. `--compact` selects
canonical one-line JSON. `--fail-on-errors` still prints the complete report
and then exits with status 2 when it contains an error-level issue. The scanner
does not follow symlink files or source directories, enforces bounded file and
aggregate input sizes, and reports unreadable, malformed UTF-8/JSON, unsafe,
or oversized sources instead of modifying or skipping them silently.

The schema-v1 report has this stable top-level shape:

```json
{
  "schemaVersion": 1,
  "kind": "legacy_composite_migration_inventory",
  "mode": "read_only_dry_run",
  "boundary": {
    "readsSavedWorkflowJson": true,
    "readsReusableBlockJson": true,
    "writesFiles": false,
    "convertsRecords": false,
    "deletesRecords": false,
    "mergesRecords": false,
    "authorizesRecovery": false,
    "containsPromptAndParameterValues": true
  },
  "sourceLayout": {
    "workflows": "user-workflows/*.json",
    "reusableBlocks": "studio/blocks/*.json"
  },
  "summary": {},
  "workflows": [],
  "reusableBlocks": [],
  "issues": [],
  "reportHash": "sha256:..."
}
```

Workflow entries classify legacy Cluster roots and their derived children, V1
`userBlockSnapshot` roots and children, already-V2 roots and projections, and
ambiguous mixed-authority nodes. Each composite inventory includes its root
and source IDs, canvas position/size, persisted presentation, prompts and
parameters, instance overrides/values, public ports and bindings, derived
child IDs, and external edges. Reusable records classify V1 and V2 definitions
and include their source ancestry, graph IDs/counts, declared ports, prompts,
parameters, and strict V2 validation result. Source SHA-256 values and a
deterministic report hash make two scans comparable without adding a mutable
timestamp.

Definition resolution against the reusable User Node store is informational.
Embedded workflow snapshots remain authoritative for later migration, and
registered catalog definitions are deliberately reported as external because
this scanner does not read the live catalog. Duplicate IDs, missing or
mismatched owners/endpoints/definitions, invalid V2 records, nested composites,
and mixed authorities produce explicit issues for human review.

The report contains exact persisted prompts and parameter values and therefore
may be sensitive. Redirecting stdout creates an operator-managed plaintext
copy; the tool itself accepts no report destination and writes no application
file. A clean inventory is not a backup, a conversion plan, execution evidence,
or authorization to recover/delete/merge anything.

The separately implemented backend preview defaults to
`GET /studio/composite-migrations/preview`. It computes a deterministic
`migrationId` and `planHash` over exact source and proposed-target hashes.
Preview remains read-only. `POST /studio/composite-migrations/apply` requires
that exact identity, literal `APPLY_BLOCK_V2_MIGRATION` confirmation, and an
explicit `allowBlockedCandidates` choice when the plan includes records that
cannot be converted safely. It converts V1 reusable definitions and embedded
V1 workflow instances only. Exact original bytes are stored below
`studio/composite-migrations/{migration_id}/backups` before replacement.

Registered Clusters remain blocked in the default GET. The additive
`POST /studio/composite-migrations/preview` accepts one strict
`registered_cluster_v2_compiler_supplement` and is still read-only. Each
compiler output is tied to an exact workflow source SHA-256 and legacy
composite hash; embeds a backend-validated `BlockInstanceV2`; matches the
backend-pinned admission's public content hash and canonical SHA-256; and
contains exhaustive one-to-one projection-node, public-port, persisted-value,
preview-output, and absorbed-internal-edge receipts. Durable input/control
values preserve the root and child prompts/parameters exactly. Volatile
`display: "output"` parameters are never treated as durable values; a non-null
media reference must instead map exactly to a matching V2 `previewState`.
The conversion also preserves the root ID, position, size, expanded state,
internal layout, external edge IDs, and outside endpoints. Missing, stale,
duplicate, structurally forked, unpinned, ambiguous, or value-changing output
remains blocked without affecting V1 or other exact conversions. Apply must
resend the same supplement so recomputing the plan fails closed before writes
if the compiler output changed. Multiple exact Cluster conversions in one
workflow are accumulated into one proposed target and therefore one atomic
workflow replacement and exact-byte backup. An invalid sibling remains visible
and byte-for-byte unchanged; an exact sibling may be included in that same
atomic target only after the operator explicitly allows blocked candidates to
remain.

Status/list endpoints report applied, rolled-back, interrupted, and conflicting
targets. `POST /studio/composite-migrations/{migration_id}/rollback` requires
literal `ROLLBACK_BLOCK_V2_MIGRATION`, verifies every backup/current hash, and
refuses independently changed or missing files. Apply and rollback are
idempotent; a rolled-back plan can be explicitly reapplied. There is no force,
delete, rename, merge, or implicit resume operation.

Setup → Advanced diagnostics now provides the default client preview and
recovery surface. Its strict parser rejects unknown/missing fields,
inconsistent counters, unsafe target paths, stale identities, invalid hashes,
and conflicting journal actions before rendering them. The review dialog lists
every exact target and blocker path. Apply requires the full literal plus a
separate `allowBlockedCandidates` opt-in when needed; rollback refreshes the
exact journal and requires its own full literal. `403`, `409`, and backend
errors stay visible, and both preview and journals refresh after success.
Nothing auto-applies. Mocked component/API tests exercise these controls
without applying against real user data. Backend compiler-supplement preview,
apply, byte-backup, rollback, idempotency, and HTTP tests are complete and use
only temporary fixtures. The client now offers a visible **Generate exact
supplement** action for exact-current registered receipts. It refreshes the
backend-owned source/composite hashes, reads the raw saved workflow, invokes
the same hidden registered V2 compiler/finalizer as insertion, removes all
transients, populates inspectable JSON, and POSTs the exact parsed object for a
final read-only preview. Apply reuses that identical object. Generation exposes
progress/cancel state, aborts on workflow replacement, and warns that dirty
tabs are not part of the saved backend bytes. The paste path remains available
for externally produced exact evidence. A disposable mocked browser lifecycle
proves Generate → Preview → Apply → List/Status → Rollback without writing a
real store.

The supported generator remains fail-closed for missing execution receipts and
historical manifest hashes without checked-in review authority. A strict
semantic-equivalence receipt can now name one exact destination compiler route;
the client validates the previewed receipt metadata, compiles through a detached
destination-compatible root, and references only its ID/hash. The backend
reloads the checked-in receipt, checks both complete identities plus exact V2
graph/interface pins, and repeats every instance-preservation check. Matching
only by definition/admission name is forbidden. The semantic-equivalence
ledger currently has zero receipts. A separate exact-body ledger now registers
six archived manifest/admission/Studio tuples covering 42 instances in the
read-only recovery audit and retains one HunyuanVideo 1.5 body as non-
convertible because its historical execution admission is absent. These bodies
are evidence, not compiler mappings. The separately reviewed mapping ledger now
covers 15 instances across three identities; 339 instances across 97 identities
remain blocked. All workflow bytes remain unchanged until exact supplement
preview and explicit Apply. The inventory CLI remains read-only and is not
conversion authority.

A separate recovery-audit-only lane now preserves the evidence that can be
recovered safely without weakening that boundary. Two byte-pinned, ignored
frontend captures yield 18 strict Studio execution-spec bodies that match 20
historical manifest identities, 21 manifest/admission/Studio-spec tuples, and
65 instances. The checked-in ledger contains only those public bodies, source
byte receipts, and collision-resistant canonical body hashes; the extractor
never ingests workflow paths, instance IDs, prompts, parameter values, or full
browser state. A schema-v4 recovery audit and
`GET /studio/composite-migrations/recovery-audit` expose the recovered body and
an empty/manual partial-review status alongside the still-missing manifest,
interface, block hierarchy, artifact authority, compiler mapping, and semantic
equivalence. Neither `verified_partial_evidence` nor `rejected_evidence` is
accepted by preview, conversion, compiler supplementation, or execution.

### Legacy registered Cluster migration

- Convert the wrapper to the common `block` canvas representation while
  preserving its canvas ID.
- Resolve the exact registered definition and embed a V2 snapshot.
- Map legacy controls and explicit Cluster ports by stable binding, not label.
- For a collapsed root whose persisted port/preview metadata references a
  deliberately absent execution child, resolve only the deterministic
  historical child ID derived from an exact unique V2 `semanticRole`; reject
  missing, duplicate, or conflicting roles without inference.
- Preserve instance values, external edges, position, dimensions, preview, and
  execution provenance.
- Preserve internal layout when available; otherwise use deterministic default
  layout without affecting graph hashes.
- If revision or content hash cannot be resolved, retain a visible
  migration-required block. Never substitute current defaults.

### Legacy Cluster-derived User Node migration

- Compare its effective graph and values with its recorded registered parent.
- When it is semantically unchanged, offer **Restore registered source** while
  preserving its instance values, preview, layout, and external edges.
- When it is structurally changed, migrate it as a workflow-owned or
  user-owned V2 block with source ancestry and its existing explicit interface.
- Never recalculate its interface during migration.
- Audit test-generated reusable definitions separately. Deletion requires a
  reviewed list and explicit approval.

### Compatibility window

- Readers support V1 and V2 during one release window.
- All new writes use V2 after the migration gate is enabled.
- A legacy export remains importable through the V1-to-V2 migrator.
- Removing V1 support requires fixture coverage and a separately documented
  release decision.

## Implementation sequence

### Phase 0 — containment and golden failures (2–4 hours)

- Prevent moving an already-owned Cluster child from triggering conversion or
  `/studio/blocks` persistence.
- Add the exact two-Qwen-instance regression described below and make it fail
  before broader refactoring.
- Stop automatic reusable-definition creation on structural edits.

### Phase 1 — V2 contracts and canonicalization (6–10 hours)

- Add runtime validators and TypeScript types for `BlockDefinitionV2`,
  `BlockInstanceV2`, boundaries, controls, source, preview, and authority.
- Add canonical hashing and explicit missing-versus-empty value handling.
- Create adapters from current registered Cluster definitions and current User
  Node definitions into the common schema.
- Add JSON round-trip and invalid-contract tests.

### Phase 2 — shared renderer and store actions (8–12 hours)

- Add the pure V2 runtime projector/reducer and persistence canonicalizer
  before routing any production definition through it.
- Route every composite through the common canvas type and renderer.
- Make dimensions, actions, controls, previews, and port trays capability-
  driven rather than source-kind-driven.
- Keep body-control and connector projections separate even when they share a
  logical ID and binding.
- Store internal layout overrides by stable internal node ID.
- Remove renderer-specific connection and serialization branches.

### Phase 3 — copy-on-write structure and interface editing (6–10 hours)

- Classify layout versus topology gestures correctly.
- Apply structural mutations to instance `effectiveGraph` without replacing
  the wrapper.
- Preserve explicit boundaries and existing external edges.
- Add the instance effective-interface contract, migration, explicit Configure
  interface operation, and edge-impact preview before enabling its capability.
- Connect the three reusable save choices to V2 definitions.

### Phase 4 — authority and execution integration (4–8 hours)

- Add and synchronize the additive multi-target control/public-input binding
  contract used when one reviewed logical value fans out to multiple ordinary
  graph fields; validate every target and apply one instance value/edge to all
  of them.
- Calculate reviewed/Auto authority from definition, effective graph,
  execution parameters, immutable artifacts, and admission.
- Keep non-Auto graph execution independent of catalog publication authority.
- Reuse the same compiled graph for collapsed and expanded views.
- Show precise preparation, dependency, resource, and worker-failure issues.

### Phase 5 — migration and recovery (6–12 hours)

- [x] The deterministic read-only inventory/dry-run report is implemented.
- [x] Backend V1-to-V2 readers and a separately authorized conversion preview
      are implemented.
- [x] Safe V1 workflow migration preserves canvas IDs, values, explicit ports,
      internal layout/effective topology, external edge IDs, and a materialized
      effective interface with synchronized base/effective hashes.
- [x] Exact byte backups, stale-plan detection, idempotent apply, recovery
      status, fail-closed rollback, and reapply are implemented and tested.
- [x] Connect the preview/recovery UI with literal-confirmation apply/rollback,
      exact target/blocker display, visible conflicts, and no automatic writes.
- [x] Connect exact registered-catalog compiler output to a strict backend
      supplement preview/apply path with source/pin/mapping validation, exact
      backups, rollback, idempotency, and preserved ambiguous records.
- [x] Accept, strictly parse, preview, review, and reuse an exact
      compiler-produced supplement in the visible migration UI; never send or
      apply it automatically.
- [x] Generate exact-current compiler supplements through a visible reviewed
      action with saved-byte input, progress/cancel, inspectable JSON, final
      backend re-preview, identical-object Apply, and disposable browser
      generate/apply/list/rollback coverage.
- [x] Register the six recovered archived definitions with exact historical
      admissions in the read-only recovery audit; retain the seventh body-only
      record as non-convertible and never infer compatibility from a current
      definition/admission name.
- [x] Add the strict historical compiler-mapping ledger, loader, compiler-
      supplement reference, preview/apply validation, journal evidence, exact
      backup/refresh/rollback path, tamper rejection, and visible browser
      lifecycle. Three independently compiled destination mappings are now
      sealed for Wan 2.2 TI2V 5B, MiniMax Music 3, and Transformers CTC speech
      recognition. They cover 15 saved historical instances and make those
      instances eligible for exact compiler-supplement preview; they still do
      not mutate a workflow without literal Apply authority.
- [ ] Isolate and seal Qwen plus the two SDXL destination mappings. Historical
      instance compilation currently publishes a different dynamic schema than
      an independent current registered insertion, so those mappings remain
      blocked rather than letting historical values redefine the destination
      BlockDefinitionV2. HunyuanVideo 1.5 remains body-only and cannot receive
      a mapping without its missing historical execution admission.
      The principled implementation path is to compile the destination once
      from the exact current registered route and its current default binding
      form, validate that immutable definition against all destination pins,
      and only then map historical values into declared boundary/control IDs on
      `BlockInstanceV2.values`. Historical values must never participate in
      destination dynamic-field publication. Unknown values, values requiring
      a graph-parameter rewrite, type-incompatible values, and values whose
      target is absent from the frozen current interface must remain blocked.
      The backend now enforces this final boundary explicitly for archived
      compiler mappings; focused tests prove graph-target rejection, unchanged
      destination definition/effective graph, cross-workflow isolation, exact
      backup, and byte-identical rollback. Qwen/SDXL can be sealed only after a
      clean-current compiler produces the reviewed definition and a separate
      value adapter proves every retained historical source maps to that frozen
      interface without schema publication.
- [x] Verify the complete browser restart and mixed V1/V2 workspace behavior.
      The mocked browser fixture persists one V1 User Node and one V2 User
      Node in the same workflow, reloads the full application, proves that
      neither authority cross-converts, checks both prompt values and exact V2
      content identity, and expands both through the shared canvas entry.

### Phase 6 — real frontend qualification and release regression (6–10 hours)

- [x] Run the bounded Qwen image golden flow through the real frontend and
      backend in Expert and Auto modes.
- [x] Qualify one current V2 video and one current V2 audio registered Block
      through the same real frontend lifecycle. Existing pre-V2 or unrelated
      media evidence does not satisfy this gate.
- [x] Complete the final post-change client checks/build/bundle, mocked browser
      suite, clean-browser smoke, and Git/evidence audit. On 2026-09-02 the
      complete `npm run check` gate, 124-case mocked Studio suite, two-case
      shared-control suite, focused live-backend Graph Fix smoke, and direct
      production-build clean-browser reload all passed. The two worktrees have
      414 source/evidence status entries rather than generated-test explosions;
      no model/media/archive/test-report/temp artifact is unignored, and both
      worktrees pass `git diff --check`. The combined V2 contract gate, full
      backend suite, migration UI gate, and route-ledger validations have also
      passed independently. Publication readiness remains a separate evidence
      gate and is not implied by this regression result.

Estimated implementation time is 4–6 focused working days, assuming migration
does not uncover ambiguous user-created legacy records. A Qwen-only demo gate
should be reachable in approximately 1.5–2 focused days.

### Phase 7 — generic registered route sets without fake generic definitions

The composite engine is source-neutral, but a reviewed execution definition is
necessarily exact to one pipeline class, workflow, admission, repository, and
immutable revision. A Qwen loader value therefore must not be edited into a
FLUX loader inside the same definition: doing that would retain stale graph,
interface, artifact, resource, and Auto authority.

The generic user experience is an instance-owned registered route set around
one active exact `BlockDefinitionV2`:

- [x] Add a strict optional `routeSelection` contract to `BlockInstanceV2` and
      synchronized client/backend validators. It records the route-set ID,
      selected route key, and bounded inactive per-route drafts; drafts exclude
      instance IDs, previews, tasks, authorities, and recursive route sets.
- [x] Register the initial Diffusers text-to-image route set with exact Qwen
      Image, FLUX, and Stable Diffusion XL routes. Route-set membership is
      catalog metadata and must never weaken the existing per-route
      hash/admission checks.
- [x] Extend registered generic selection only where the reviewed definitions
      retain the same task semantics and a stable compatible public output.
      The current exact sets are text-to-image (3 routes), image-to-image (4),
      instruction edit (5), inpainting (3), control image (2), text-to-video
      (9), and image-to-video (7). Every member resolves through the immutable
      definition/admission registry; no model name is substituted into another
      model's graph.
- [x] Compile the destination route in the existing transient exact compiler,
      preflight every connected public port, and atomically replace only the
      active definition/graph/interface/value projection while preserving the
      root instance ID, canvas position, size, and compatible external edges.
- [x] Preserve each inactive route's values, effective graph/interface, and
      internal layout. A first visit carries only explicitly portable semantic
      values (`prompt` and `seed` initially); revisiting a route restores its
      prior draft byte-for-byte. Switching clears previews and all execution,
      resource, Auto, and publication authority.
- [x] Expose the selector in the shared `BlockNodeV2` renderer for both
      registered and customized workflow instances. Customized instances must
      offer keep-as-workflow-draft, save-active-route-as-new-User-Node, or
      cancel; no registered definition is overwritten implicitly.
- [x] Cover Qwen → FLUX → Qwen → SDXL → Qwen restoration, save/refresh/app restart,
      multi-instance isolation, one-step undo/redo, compatible-edge retention,
      incompatible-port refusal, exact active-route execution export, and real
      frontend generation for the selected Qwen, FLUX, and SDXL routes.

Completed on 2026-09-03: the shared renderer exposes the exact Qwen Image 2512,
FLUX.1 Dev, and Stable Diffusion XL 1.0 routes. A customized active route can remain as an inactive
workflow draft or be saved as a new User Node before the same root switches;
the latter passed a live backend write/delete lifecycle without rebasing the
workflow insertion during the save. The strengthened browser proof covers two
independent roots; Qwen → FLUX → Qwen → SDXL → Qwen draft restoration;
undo/redo; save/refresh; and a fresh-browser reopen after a full backend
restart. Execution expansion and the submitted executable node
map contain only the selected route; the full workflow snapshot intentionally
retains inactive drafts as non-executable provenance. Current-V2 Qwen, FLUX,
and SDXL routes completed real visible-frontend generation with pinned Hub
artifacts. The FLUX route-set proof used BF16, no quantization, model CPU
offload, 768×768, 20 steps, and seed 271828; its fresh review evidence is under
`review-pending/qwen-flux-generic-v1-20260903/qwen-flux-route-switch-v1/`.
The SDXL proof used FP16, no quantization, model CPU offload, 1024×1024,
30 steps, guidance 5, and seed 161803; its isolated current-run evidence is
under
`review-pending/qwen-flux-sdxl-generic-v1-20260903/qwen-flux-sdxl-route-switch-v1/`.
The exact checked production build was then copied into the backend `web/`
tree. A clean headless browser opened the retained backend workflow, rendered
both V2 roots, and exercised the deployed route selector without a JavaScript
page error. The retained manual-demo workflow is
`O6Xsgtn29NIMNuBP9zi8l` (`Demo — Generic Qwen / FLUX Image Block`).

Expanded on 2026-09-03: exact compatible model selection now also covers
Qwen/FLUX/Anima/SDXL image-to-image; Qwen Edit/Edit Plus/FLUX Kontext/FLUX.2
Klein/Klein Base instruction editing; Qwen/Qwen Edit/SDXL inpainting;
Qwen/SDXL control image; and Wan 2.1/Wan 2.2/LTX/LTX-2/HunyuanVideo 1.5/
Helios/Helios Pyramid/Helios Pyramid Distilled/Cosmos 3 Omni text-to-video.
Image-to-video covers Cosmos 3 Distilled/Omni, HunyuanVideo 1.5, LTX/LTX-2,
and Wan 2.1/2.2. The three current Helios image-to-video definitions are
excluded because their reviewed public boundaries expose only source height
and width rather than a video output; they must be corrected and recompiled
before they can safely join a generic video-output set.
The registry audit proves that each entry maps to one exact registered route,
that a route is not ambiguously assigned to multiple sets, and that every set
retains a stable shared output type. A production-browser test inserts a fresh
Block for all seven sets and verifies every visible option. A second browser
test performs real Qwen Edit → FLUX Kontext → Qwen Edit and Wan → LTX →
Wan round trips while preserving the workflow prompt.

Exact inactive drafts whose compiled definition ID, content hash, and canonical
SHA-256 still match the current registry are now restored without rebuilding
the hidden transient Modular graph. The restored value is still passed through
the complete `BlockInstanceV2` validator, receives a fresh execution-neutral
instance projection, and has previews and authorities cleared. Missing or stale
drafts continue through the full compiler/rebase path. The three-route lifecycle
fell from 44.5 seconds to 33.6 seconds, while the new edit and video round trips
complete together in 16.4 seconds on the qualification host.

## Required tests and acceptance evidence

### Primary two-instance browser regression

1. Insert two identical Qwen registered blocks.
2. Assert byte-equivalent V2 definitions and equal rendered controls, actions,
   port IDs/types/order, size behavior, and starter prompt.
3. Run the left instance and open its generated asset.
4. Expand it and move one existing owned internal node.
5. Collapse it.
6. Assert it is still the same `block` instance with the same definition
   reference, controls, values, ports, name, authority, and external edges.
7. Assert only preview/run state and internal layout may differ.
8. Assert no reusable User Node API request occurred.
9. Save, refresh, reopen, and repeat the equality assertions.
10. Run again through the visible frontend.

### Structural customization tests

- Adopt a compatible ordinary node, reconnect it, collapse, Save, refresh,
  re-expand, and Run.
- Preserve wrapper ID, public ports, external connections, values, layout, and
  preview while authority changes predictably.
- Reject invalid type, state, component, and container connections with a
  useful message.
- Undo/redo movement, adoption, reconnection, deletion, and replacement.
- Verify all three save choices and multi-instance isolation.
- Confirm a registered definition cannot be overwritten.

### Schema and migration tests

- Valid and invalid V2 fixtures for every source kind and boundary mode.
- Stable canonical hashes across JSON key order and presentation changes.
- Explicit empty values survive round trip and are not replaced by defaults.
- Stable port identities preserve external edges across renderer migration.
- V1 Cluster, V1 User Node, modified derived node, missing source, and stale
  revision fixtures.
- Read-only inventory output and `reportHash` are repeatable for identical
  inputs; caller-owned values and scanned workflow/block bytes and mtimes remain
  unchanged.
- A future conversion preview is deterministic and a future authorized
  migrator is idempotent; applying it twice produces no additional changes.

### Cross-modality tests

- One registered image, video, and audio block use the same composite renderer
  contract.
- Each exposes a task-appropriate explicit boundary and creator input examples.
- Each survives edit, Save, refresh, expansion, layout movement, collapse, and
  real frontend execution without value reset.

## Definition maintenance requirements

Any change to `BlockDefinitionV2` or one of its nested contracts MUST include,
in the same change:

1. the client type and strict validator in
   `MoDiff-client/src/studio/blockSchemaV2.ts`;
2. the synchronized backend type and strict reusable-definition validator in
   `MoDiff/modiff/block_definition_v2.py`;
3. matching client and backend canonical serialization and hashing updates;
4. a schema fixture for each affected source/boundary variant;
5. an explicit migration or a proof that the change is backward-compatible;
   when V1 conversion semantics are affected, update both
   `migrateUserBlockDefinitionV1` in the client and
   `migrate_user_block_definition_v1` in
   `MoDiff/modiff/composite_migration.py` and prove the same graph/definition
   hashes;
6. registered-source adapter changes in
   `MoDiff-client/src/studio/registeredBlockAdapterV2.ts` when the field is
   produced from the Hugging Face catalog;
7. updates to this document and linked API documentation; and
8. round-trip, cross-runtime hash, migration, and browser coverage appropriate
   to the field.

The minimum synchronized test set is:

- `MoDiff/scripts/verify-block-v2-contract.sh` is the single local gate for
  the client schema, registered adapter, runtime, renderer, capability policy,
  reusable-definition persistence, backend definition-store, workflow-store,
  migration inventory, conversion, transaction, recovery, and HTTP suites.
  `MODIFF_CLIENT_ROOT` may point it at a non-sibling client checkout;

- `MoDiff-client/scripts/block-schema-v2.test.mjs` for strict definition and
  instance validation, hash stability, explicit-empty values, V1 reading, and
  source/boundary variants, including cross-direction public-port uniqueness
  and deterministic effective-interface migration/hash validation. The
  current client schema gate contains ten focused tests, including exact
  client/backend V1 effective-interface hash parity and cross-runtime mirrored
  control/public-input hash parity;
- `MoDiff/tests/test_composite_migration.py` for the synchronized V1 reader,
  workflow identity/value/interface/layout/edge preservation, exact backups,
  stale-plan rejection, idempotency, and fail-closed rollback whenever schema
  or migration semantics change;
- `MoDiff-client/scripts/composite-migration-ui.test.mjs` for strict preview,
  journal, status, and mutation-response validation; exact apply/rollback
  request authority; visible `403`/`409` errors; exact-path rendering; and
  confirmation/blocked-candidate gates against mocked APIs only;
- `MoDiff-client/scripts/registered-block-v2-adapter.test.mjs` for stable
  semantic graph IDs, exact registered provenance/interface compilation, and
  separation of definition state from instance values/layout/run state;
- `MoDiff-client/scripts/registered-block-v2-route-audit.test.mjs` for the
  current live `/huggingface/node-library` schema-v6 inventory. It compiles
  every graph-qualified admission against its exact Studio receipt and live
  node fields, executes every declared dynamic field action in declaration
  order using only sealed catalog/default inputs, replays the browser's exact
  shallow publication merge, repeats each witness to reject nondeterminism,
  requires every compiler success to have one explicit route,
  pins the current 90/90 split (82 Diffusers, eight Transformers), verifies
  both the public definition content hash and canonical-definition SHA-256,
  and rejects catalog-hash, revision, artifact, spec, admission, action, and
  boundary drift;
- `MoDiff-client/scripts/block-runtime-v2.test.mjs` for single-authority root
  persistence, separate control/connector projections, explicit-empty values,
  internal layout, copy-on-write mutations, authority invalidation, and
  execution expansion. It must also fail closed for spoofed, orphan, or stale
  projections; missing control, sealed-control, boundary, or preview bindings;
  incompatible internal handles/types; unsupported node types; duplicate
  identities; and projection-ID collisions. Its current 24 focused tests also
  cover mirrored value/edge fan-out, root-to-root fan-out, identical
  collapsed/expanded execution expansion, and fail-closed
  readiness/export behavior. A runtime-only green test does not imply that
  broad registered catalog insertion or Auto has been routed through V2;
- `MoDiff-client/scripts/composite-block-capabilities-v2.test.mjs` when
  ownership, permission, save-action, or effective-interface behavior changes.
  `configureInterface` resolves only from a valid definition/instance pair and
  an explicit workspace interface-edit permission;
- `MoDiff-client/scripts/block-persistence-v2.test.mjs` for all three save
  choices, registered-update rejection, exact reusable graph/interface/default
  preservation, boundary-only value fail-closed behavior, API rollback,
  semantic save races, sibling/open-workflow isolation, and independent User
  Node reinsertion. Its current focused gate contains nine tests; and
- `MoDiff/tests/test_studio_blocks.py` for backend canonical hashes, fail-
  closed V2 validation, exact round-trip persistence, registered-source write
  rejection, V1 compatibility, and cross-direction public-port uniqueness. The
  current backend contract gate contains ten tests with 26 subtests.

Changing only one canonicalizer is a contract defect even when its local tests
pass. Before merging a V2 field or hash change, at least one shared fixture
MUST be hashed by both runtimes and produce the same exact strings.

An instance-only `BlockInstanceV2` change does not require inventing a backend
definition field. It MUST still update the client type, strict instance
validator, workflow serialization/migration, schema fixtures, and lifecycle
tests together. If backend code starts interpreting that instance field, its
validator and tests join the same synchronized-change rule.

Field removal or semantic reuse requires a new schema version. Optional fields
may be added within V2 only when old readers fail closed or ignore them without
changing execution, interface, values, or authority. A field name MUST NOT be
reused with a new meaning.

Definition generators MUST validate before publication and produce stable
canonical hashes. Client code MUST consume declared definitions; it must not
contain model-family branches that reconstruct boundaries or controls.

## Status checklist

### Contract and documentation

- [x] Corrected one-composite product contract documented.
- [x] `BlockDefinitionV2` and `BlockInstanceV2` semantic contracts documented.
- [x] Boundary, provenance, authority, migration, and maintenance rules
      documented.
- [x] Legacy visual/demo plans marked as superseded where they conflict.

### Functional implementation

- [x] Moving an owned registered-block child is layout-only and cannot create
      or persist a User Node.
- [x] Mocked two-instance Qwen regression preserves root identity, controls,
      ports, values, renderer class, and performs zero `/studio/blocks` writes.
- [x] Common client `BlockDefinitionV2`/`BlockInstanceV2` types, strict
      validators, canonical hashes, explicit-empty value behavior, and V1
      User Node reader are implemented and covered by focused tests.
- [x] Backend `BlockDefinitionV2` types, strict validation, canonical hashes,
      user-owned V2 `/studio/blocks` round trips, registered-source write
      rejection, and V1 compatibility are implemented and tested.
- [x] The client and backend parity fixture asserts the same fixed
      `block-graph-v2-d23b144c` and `block-definition-v2-dde5d4d9` hashes.
- [x] The generic registered Diffusers/Transformers adapter produces stable
      semantic graph IDs, explicit ordered boundaries and controls, creator
      suggestions, preview bindings, immutable provenance, and independent
      instance values/layout without hashing volatile run or UI state.
- [x] The source-neutral capability resolver is implemented and fail-closed;
      registered definitions cannot be updated, while structural editing and
      Configure Interface are permission-gated equally for registered and
      user-owned workflow instances.
- [x] The pure source-neutral V2 runtime/projector and persistence canonicalizer
      create one `block` root from `BlockInstanceV2`, keep controls separate
      from connector trays, project ordinary internal nodes and links, preserve
      explicit-empty values and presentation-only layout, and canonicalize an
      already-V2 workflow without persisting derived children. This slice is
      implemented with fail-closed projection, binding, type, handle, duplicate,
      and collision validation and is covered by focused tests. All 90 exact
      registered admissions currently use this route.
- [x] Already-V2 roots use React Flow `type: "block"`, the `BlockNode` entry,
      and the same `BlockNodeFrame` used by legacy User Nodes. Focused tests
      cover action placement, resize minimums, controls, explicit connector
      types, independent values, expansion, parent-relative layout,
      duplication, and non-history refresh rematerialization. All 90 exact
      registered admissions named by the live route table are routed to this
      branch.
- [x] The expanded V2 wrapper derives its temporary canvas bounds from every
      owned ordinary child's persisted layout and measured dimensions while
      retaining the exact durable collapsed size. A focused runtime invariant
      and the real-backend Qwen text-to-image browser lifecycle assert that
      all five Modular Diffusers children remain inside the root before and
      after moving the prompt node and after Save/refresh/re-expansion. The
      same lifecycle proves unchanged definition/interface/value authority,
      collapsed/expanded API export parity, independent sibling state, and
      successful registered-Block and saved-User-Node execution.
- [x] Every one of the ten visible Qwen catalog Blocks passes a real-browser
      empty-graph insertion/expansion contract: the library row is draggable,
      insertion creates one V2 root, every projected Modular Diffusers node is
      visually contained by that root, and collapsing leaves the serialized
      instance byte-identical. The same family gate requires at least one
      explicit public input and output, a non-empty connected internal graph,
      and exact preservation of the publisher-sourced starter prompt whenever
      the upstream definition provides one. A definition with multiple reviewed admissions
      uses its first explicitly ordered admission as the deterministic empty-
      graph default; an explicit matching Studio task/model may select another
      admission. Selection still fails closed unless the exact registered V2
      route, immutable artifact, and execution receipt match.
- [x] Fresh registered compiler values are now explicitly marked as the
      insertion baseline rather than workflow customization. Runtime edits use
      the same default comparison, so changing only a prompt changes only that
      value and the customization marker, while restoring the reviewed value
      can make the instance clean again. Qwen Edit Plus exposes separate
      single-image and multi-reference choices in the shared instruction-edit
      route selector; exact admission lookup no longer assumes one admission
      per catalog definition. The complete ten-entry real-browser family gate
      exercises the multi-reference switch and asserts every new root begins
      unchanged.
- [x] Already-V2 public inputs and outputs are resolved from the explicit V2
      boundary throughout connection lookup, type/color rendering, pane drop,
      connect/remove, status, and signal paths without copying socket state
      into root `params`. External edges stay on the durable root; projected
      children accept only same-instance internal connections. Internal edge
      additions/removals update that instance's `effectiveGraph` copy-on-write.
- [x] API export, run readiness, and run-target resolution expand already-V2
      roots into the same concrete graph in collapsed and expanded views. A
      malformed V2 graph produces a blocking
      `composite_execution_graph_invalid` issue even in Expert mode, and the
      wrapper/instance contract is never submitted as an executable node.
- [x] Concrete V2 WebSocket outputs and progress resolve deterministically to
      the owning collapsed root and declared preview binding. Generated media,
      task ID, and status update only that instance's `previewStates`; root
      `params`, definition state, sibling instances, and undo history remain
      unchanged. The focused renderer and run-coordinator gates cover this
      route.
- [x] The exact Qwen Image text-to-image catalog admission now fetches one
      build-time generated, hash-pinned V2 entry and atomically returns one
      source-neutral `block` root. Fresh insertion creates no compiler nodes,
      runs no field actions, imports no optional runtime, accesses no Hub, and
      loads no model. Source identity and canonical SHA-256 are checked by
      both runtimes before graph mutation. The same generated-catalog gate now
      admits all 90 pinned routes (82 Diffusers and eight Transformers). The
      hidden dynamic-field compiler remains only as a bounded, receipt-bound
      V1 recovery fallback; timeout, stale-hash, and workflow-replacement tests
      prove atomic cleanup with no partial graph mutation.
- [x] The exact Qwen Image text-to-image mocked browser lifecycle covers
      catalog drag, one durable V2 root, creator prompt and explicit public
      ports, five linked ordinary internal nodes, internal layout movement,
      collapse, value edit, workflow Save/reload, stable root/definition/
      interface/value/layout authority, and zero legacy Cluster receipts or
      `/studio/blocks` writes. This is not a real-backend generation claim.
- [x] Deleting an eligible expanded V2 internal node is an atomic
      copy-on-write structural edit with incident-edge/execution-order/layout
      cleanup, authority invalidation, undo/redo, and Save/refresh
      rematerialization coverage. Deletion fails visibly and without partial
      mutation when the node still owns a public port, exposed control, or
      preview. Deleting the root still removes the whole composite.
- [x] The pure source-neutral V2 runtime can add/adopt one already-filtered
      ordinary semantic node copy-on-write with stable ID, nested-composite,
      duplicate, reserved-projection, execution-order, layout, graph-hash, and
      authority-invalidation checks. One safe store/canvas gesture now adopts
      a disconnected top-level ordinary node into an expanded V2 Block with
      exact relative layout, one-step undo/redo, canonical Save/refresh
      rematerialization, unchanged explicit ports, and visible atomic
      rejection. The renderer/store suite has 16 focused tests and a mocked
      real-pointer browser lifecycle covers both success and connected-node
      rejection. A second mocked real-pointer registered-V2 lifecycle covers
      compatible replacement, incompatible atomic rejection, declared-public-
      edge adoption, Configure Interface edge-impact rejection and explicit
      mirror fan-out, internal and external reconnection, move-out through an
      existing public port, undo/redo, collapse, Save, refresh, re-expansion,
      and zero reusable User Node writes. Move-out now records and excludes
      its source Block from later target scans in the same drag-stop gesture,
      preventing the ordinary materialized node from being immediately
      re-adopted while still permitting a direct drop into another Block.
- [x] The legacy two-Qwen containment regression covers independent insertions,
      one-instance prompt editing, child-layout movement, Save, refresh,
      rematerialization, and zero `/studio/blocks` writes. It is migration
      safety evidence, not shared-renderer completion.
- [x] Complete the current Qwen golden lifecycle as two bounded proofs. The
      real-backend frontend proof covers two-instance isolation, edited values,
      internal layout persistence, collapsed/expanded concrete-graph parity,
      Expert execution/output viewing, and one exact Auto execution. The mocked
      real-pointer proof covers the structural-edit gestures listed above
      through the production V2 renderer/store route. This does not claim that
      the structural gesture fixture generated a real model output.
- [x] Route every admitted registered definition through V2 insertion and
      saved workflow loading instead of retaining the legacy `cluster`
      wrapper. All 90 admitted/graph-qualified routes pass the exact live
      generated-catalog gate (82 Diffusers and eight Transformers). Control/public-input
      fan-out and exact terminal-output projection use additive source-neutral
      V2 mirror contracts. Every future route MUST continue to fail closed
      until its own exact semantic boundary, fully finalized dynamic schema,
      content hash, and canonical SHA invariants pass.
- [x] Keep discovery-only or route-mismatched catalog definitions visible and
      structurally insertable through the shared V2 renderer, but without Run,
      Auto, or publication authority. Executable catalog projection still
      requires an exact registered V2 route; no new interaction can create a
      legacy `cluster` root as a fallback, and route drift leaves the
      canvas and undo history unchanged.
- [x] Bind every remaining catalog-only definition to one immutable review
      record through `huggingface-cluster-catalog-gates.v1.json`. The original
      audit covered 28 workflows across seven families. Fourteen have since
      received exact V2 route admissions; the current catalog-only remainder
      is 14 workflows across five families: Cosmos 3 (five), LTX-2.5 (four),
      Krea 2/Turbo (two), Stable Diffusion 3 (two), and Ideogram 4 (one). The
      ledger validates each evidence-file SHA-256 and unresolved artifact,
      access/license, runtime/resource, or guardrail gate. The current eligible
      route count remains zero, so none of those 14 is mislabeled draggable
      before its own exact route contract is admitted.
- [x] One canonical `block` canvas type and one shared renderer for registered
      and user-owned blocks is in production. Registered versus User Node is
      catalog ownership/provenance only; it does not select a second renderer,
      canvas type, control surface, connector path, or structural editor.
- [x] Copy-on-write structural editing preserves the wrapper: connector
      add/remove/reconnect, safe deletion, disconnected adoption, declared-
      public-edge adoption, compatible node replacement, and move-out through
      existing public ports are integrated with undo/redo and atomic failure.
      Arbitrary crossings intentionally require disconnect → move → Configure
      Interface → reconnect. Preview/sealed-owner replacement is supported only
      through the compatible identity-preserving route, so immutable bindings
      and sealed values are never rebound. The focused runtime and mocked
      browser lifecycles qualify protected replacement, atomic rejection,
      undo/redo, and Save/refresh persistence without broadening those limits.
- [x] Explicit, stable boundary preservation and Configure Interface flow use
      the validated instance `effectiveInterface`; connected-port removal,
      invalid bindings/types, sealed-control mutation, and stale hashes fail
      closed.
- [x] The three explicit V2 persistence choices are wired: workflow-only makes
      no reusable-definition write; save-as-new preserves the exact effective
      graph, explicit interface, declared current defaults, preview bindings,
      and parent provenance; and update-existing is restricted to mutable
      user-owned definitions. API failures roll back the User Node library,
      semantic save races abort target rebasing, sibling and other-open-
      workflow snapshots remain unchanged, and saved V2 User Nodes reinsert by
      click or drag as independent instances. Boundary-only valued inputs fail
      closed because `BlockPortV2` has no reusable default. Focused tests cover
      failure, race, isolation, and reinsertion semantics; a mocked browser
      lifecycle exercises all three visible choices and refreshes the updated
      instance.
- [x] Finish the primary real-backend reviewed-execution/Auto browser
      qualification for the exact current Qwen Image text-to-image route.
      Already-V2 manual execution expansion and fail-closed readiness are
      implemented. Routed registered V2 instances now request a short-lived,
      instance-local Auto receipt from the backend. The backend strictly
      validates the normalized instance, admission, immutable artifact, and
      complete resource form; recomputes graph/interface/value hashes; and
      verifies both the public V2 definition hash and a canonical-definition
      SHA-256 pin. Structural/interface customization fails Auto without
      weakening independent Expert/manual execution. The 2026-09-02 live
      browser proof now covers catalog drag, exact V2 definition pin, edited
      instance values, instance-local Auto receipt, unchanged values after
      authority, five-node concrete submission, completed backend execution,
      frontend preview update, generated image retrieval, and post-run browser
      capture. It also exposed and fixed a backend receipt timestamp that did
      not round-trip through the shared strict V2 validator. This proof is for
      the exact Qwen route; it does not claim that every registered route has
      received a real model run.
- [x] Deterministic, machine-readable legacy inventory/dry-run report for saved
      workflows and reusable User Node records. It identifies legacy Cluster,
      V1 User Node, and already-V2 authorities plus children, interfaces,
      values, topology, definition resolution, and ambiguity without modifying
      either store. Focused pure/filesystem/CLI tests pass; this is inventory
      evidence only and does not authorize migration or recovery.
- [x] The backend now has a deterministic V1 User Node definition/instance
      reader, read-only conversion preview, explicitly authorized safe-target
      application, exact byte backups, idempotency, transaction status, and
      fail-closed rollback. Root IDs, values, declared ports, internal layout/
      topology, and external edge IDs are preserved. Legacy registered
      Clusters remain blocked by default; an exact pinned catalog compiler can
      now supply a strict source-bound supplement whose V2 instance, public and
      canonical definition pins, one-to-one node mappings, explicit port
      bindings, all persisted values, absorbed projection edges, layout, and
      external edge rewrites are validated before the candidate becomes
      convertible. The backend never reconstructs execution authority from
      presentation children.
- [x] Add the migration preview/recovery UI in Setup → Advanced diagnostics:
      strict typed preview/journal/mutation parsing, exact target and blocker
      paths, literal-confirmation review dialogs, explicit blocked-candidate
      opt-in, visible `403`/`409`/backend errors, refreshed state after actions,
      and mocked tests that never mutate real user data. There is no automatic
      apply or force rollback.
- [x] Add the backend registered-Cluster compiler handoff with deterministic
      read-only preview, exact-supplement apply, byte backup, rollback, and
      tempfile-only tests. No delete, rename, or merge behavior exists.
- [x] Add visible exact-current client supplement generation, receipt review,
      final read-only submission, identical-object Apply, and recovery UI;
      keep the default preview blocked and never auto-apply.
- [x] Add the fail-closed historical-manifest semantic-equivalence mechanism:
      a checked-in, strictly hashed receipt ledger pins the complete old
      manifest/admission/Studio-spec identity and reviewed historical graph/
      interface hashes to one exact current manifest plus
      `BlockDefinitionV2` content/canonical/graph/interface hashes. The client
      shows the issuer/timestamp/notes and old-to-new hash diff in read-only
      preview, compiles only the receipt's exact destination route, and sends a
      receipt ID/hash reference. The backend re-resolves that checked-in
      authority and still requires exhaustive instance mappings, literal Apply,
      exact backup, and rollback. The ledger is intentionally empty: this adds
      no fabricated authority and converts no saved workflow.
- [ ] Add individually reviewed historical compiler mappings or compatibility
      receipts before offering generation for each legacy identity that does
      not exactly match a current route.
      The read-only 2026-09-02 audit currently finds 375 legacy Cluster
      instances across 102 manifest identities. Only 21 instances across two
      identities are exact-current. The remaining 354 instances span 100
      historical identities (343 with receipts and 11 without). Exact retained-artifact recovery found
      seven complete historical bodies. Six have exact execution-tuple
      agreement and cover 42 instances; HunyuanVideo 1.5 covers three instances
      but has no historical execution admission and remains body-only. The
      other 312 instances across 94 identities still require historical
      evidence. Three reviewed mappings now cover 15 instances across three
      archived identities, leaving 339 instances across 97 identities blocked.
      All 354 workflow instances remain byte-unchanged because mapping presence
      alone is not mutation authority. Any added mapping/receipt must
      pin the complete historical source/execution tuple and exact destination
      V2 definition/graph/interface; the associated compiler output must still
      receipt every semantic node/value/port/preview/topology mapping.
      Matching labels, model IDs, presentation children, or a current catalog
      route are not authority.
      The schema-v4 evidence audit now makes this actionable per identity and
      reports the disjoint `15 mapping-ready / 339 still blocked` aggregate: all
      375 saved roots are identity-only references, zero embed a complete
      definition, 364 retain a complete admission/Studio-spec tuple, nine roots
      retain 63 non-authoritative presentation children, and no migration
      journal/backup exists. It also distinguishes manifest/admission matching
      from exact Studio execution-tuple matching, reports the same current
      `21 / 2` on both axes, and supplies precise missing-evidence codes plus
      safe recovery/review actions without exposing private workflow data.
      Two exact local capture receipts recover 18 self-hash-valid Studio bodies
      covering 20 of those historical identities, 21 exact execution tuples,
      and 65 instances. Those bodies are checked-in partial evidence only; the
      manual-review ledger starts empty and conversion authority remains zero.
      A separate read-only audit recovered the complete self-hash-valid Qwen
      manifest body
      `sha256:06fc8c7d2ec50311261f70c6942d479f3151f61026fbcf42b582c4c180536b85`
      from retained local execution
      evidence for the two instances made historical by the current route
      reseal. Its graph, interface, artifact, admission, and Studio execution
      tuple are identical to the current route; only publication metadata and
      the resulting definition pins differ. It is an archive candidate, not
      authority: Git does not authenticate that retained body, and both legacy
      roots persist six values (`attention_kwargs`, `generator`, `latents`,
      `num_images_per_prompt`, `output_type`, and `sigmas`) that the frozen
      current V2 interface cannot represent. One root also persists a required
      null prompt. The compiler correctly refuses to discard, coerce, or hide
      those values. Recovery therefore requires an explicit reviewed dormant-
      value preservation contract or a newly reviewed V2 interface; matching
      graph/interface hashes alone is insufficient.
      The independent archived-definition ledger is collision-resistant and
      fail-closed: all seven definition, source-record, record, and ledger
      hashes are validated; only the six definitions with recovered execution
      admissions enter the read-only audit. Raw transcript records, private
      workflow values, paths, and instance IDs are not checked in.
      The generic checked-in compiler-mapping mechanism is now implemented and
      tested independently: it binds the archive record and complete historical
      manifest/admission/Studio tuple to exact destination manifest,
      BlockDefinitionV2 content/canonical, graph, and interface hashes. The
      visible generator emits only an ID/hash reference; the backend resolves
      the ledger again, rejects duplicates/drift/tampering, and revalidates the
      compiled destination before the existing literal-confirmation Apply.
      Focused backend tests cover preview, Apply, persisted refresh, rollback,
      tampering, and the Hunyuan body-only refusal; a mocked browser test covers
      generate, review, Apply, recovery listing, and rollback. Destination
      Three exact records are sealed and expose mapping authority for 15 saved
      instances. Qwen and two SDXL records remain pending dynamic-schema
      isolation; Hunyuan remains body-only.

### Qualification

- [x] Primary real Qwen two-instance Expert regression and exact single-
      instance Auto regression pass through the frontend. The preserved
      evidence lives under `review-assets/v2-live-2026-09-02/`; proof-instance
      256×256 and two-step values are workflow-local overrides and do not
      change catalog defaults.
- [x] One real current-V2 image lifecycle passes through Qwen Image text-to-
      image in both Expert and Auto modes.
- [x] The current-source Qwen golden path uses the exact pinned
      `Qwen/Qwen-Image-2512` creator prompt and negative prompt. A real frontend
      run proves two-instance isolation, parameter-only save/refresh parity,
      collapsed/expanded export parity, Cluster-to-User-Node ancestry and
      byte-identical reload, and execution of both the registered instance and
      saved User Node. A separate 1328×1328, 50-step, seed-314159 frontend run
      completed in BF16 with no quantization and no provenance blockers.
- [x] The complete ten-entry visible Qwen catalog family now has a strengthened
      real-browser lifecycle gate. Each entry is inserted into an empty graph,
      receives a workflow-local prompt edit, is saved, survives a full browser
      refresh byte-for-byte, and re-expands with every linked Modular Diffusers
      child contained by the shared Block frame. The Qwen Edit Plus entry uses
      its explicit multi-reference route in this pass. The isolated receipt is
      `review-pending/qwen-v2-family-lifecycle-20260903/qwen-v2-family-lifecycle/frontend-result.json`;
      all ten records report Save/refresh identity and post-refresh containment.
- [x] Execute the ten Qwen admissions not covered by the completed Image
      text-to-image proof through that same current-V2 frontend lifecycle:
      Image image-to-image/inpaint/three ControlNet routes, Edit image/inpaint,
      Edit Plus single/multi-reference, and Layered decomposition. Preserve
      every technical output and incremental route receipt under the isolated
      `review-pending/qwen-v2-admission-executions-20260903/` directory. Do not
      change catalog defaults; low-cost test values are workflow-instance
      overrides recorded in each receipt.
      The first current-V2 image-to-image attempt exposed and is now guarded
      against a missing execution-validation rule: an explicitly reviewed
      `media_file_path` boundary may retain its publisher media union while it
      binds to an internal file browser. Only that narrow media/file-browser
      adaptation is accepted; unrelated type mismatches still fail closed.
      The same first run exposed an intermediate Image Encode text summary
      that lexical ordering had marked as the primary preview. Run-from-Block
      now prefers a terminal declared preview, with `primary` used only as a
      terminal tie-breaker/fallback, so diagnostic branches cannot truncate
      generation.
      Completed on 2026-09-03: all ten routes have completed frontend tasks
      and retained media in
      `review-pending/qwen-v2-admission-executions-20260903/qwen-v2-admission-executions/`.
      Qwen Layered now exposes its upstream-required `resolution` as one
      explicit fan-out control bound to both prompt encoding and image
      encoding. Its Studio specification, compiled `BlockDefinitionV2`, and
      immutable client route pins were updated together and re-audited.
- [x] Repeat the structural lifecycle against the current Qwen Image V2 graph
      and retain the modified execution path. The frontend proof replaces the
      prompt node with a compatible ordinary node, verifies that the original
      node alone is removed and all effective-interface bindings follow the
      replacement, deletes and visibly reconnects its embeddings edge, saves,
      refreshes byte-identically, re-expands with contained linked children,
      and completes a real model run. Parameter-only mutation is separately
      compared against the complete instance so no unrelated field, graph,
      interface, definition, or presentation state can change. Evidence is in
      `review-pending/qwen-v2-structural-execution-20260903/qwen-v2-structural-execution/`.
- [x] Structurally customized Blocks remain ineligible for exact route,
      publication, and Auto authority, but now derive a narrow cleanup-only
      model/artifact identity from the submitted loader. This prevents a prior
      large model from remaining resident merely because customization removed
      the exact route receipt. The derived identity cannot provide a Studio
      execution receipt or Auto candidate; it can only trigger process-wide
      cache release. The real Qwen structural run exposed this on ROCm after a
      Qwen Layered run, and the corrected run completed after supervised
      cleanup with no stale model state.
- [x] The current-V2 structural golden path now also executes the modified
      effective graph rather than removing the inserted node before Run. From
      the visible frontend it inserts the registered Qwen Cluster, changes only
      workflow-local values, uses a real visible pointer drag to adopt a normal
      Data Viewer, connects it to the decoded image, saves the result as a
      user-owned `BlockDefinitionV2`,
      refreshes, and runs the complete graph. The exact run produced both the
      640×640 image and the terminal Data Viewer output. A subsequent full
      backend restart and clean browser load preserved the definition ancestry,
      graph hash, semantic connection, prompt, and parameters. The retained
      manual demo records are workflow `CjFV7JYwMYzeG8XzXaV00` and User Node
      `user-block-v2-ns8dy0yLgzsqZLZW`; fresh evidence is isolated under
      `review-pending/qwen-v2-custom-user-node-20260903/`. The verified
      production bundle was then deployed, the backend restarted, and the same
      retained workflow run from the backend-served UI as task
      `3OkNYjDNDyk-`; it reproduced both the original image and Data Viewer
      byte hashes.
- [x] Workflow backend reconciliation now clears a delayed dirty marker only
      when the complete current document is byte-signature-equivalent to the
      backend-owned document. A dirty-only state transition triggers the sync
      loop, while any genuinely newer local content remains dirty. This closes
      the explicit-save/autosave timer race encountered by the real structural
      lifecycle without weakening conflict protection.
- [x] Random-field controls project their persisted `{ value, isRandom }`
      payload into the numeric execution seed and explicit randomization flag;
      this is covered independently of the Qwen browser run so presentation
      metadata cannot leak into backend parameter types.
- [x] Graph Fix detects invalid V2 effective structure through the same
      execution validator used by Run, highlights the owning Block, and offers
      an explicit reviewed-structure recovery. Focused tests prove registered
      nodes/links are restored, compatible custom additions survive, and valid
      customized Blocks receive no structural issue. A visible mocked-browser
      lifecycle additionally corrupts a current registered Qwen instance,
      requires the explicit recovery choice, retains a compatible custom node,
      records one undo step, saves, refreshes, and revalidates the repaired
      execution graph.
- [x] Expanded V2 projection reserves a header-safe inset for older reviewed
      execution layouts while preserving their relative coordinates and
      immutable semantic definition. A real-backend Helios Pyramid Distilled
      test proves the first ordinary Modular node no longer overlaps or
      intercepts the shared Block action bar after Save and refresh; the same
      test checks all 95 reviewed Diffusers definitions remain draggable and
      that collapsed/expanded concrete execution is identical.
- [x] Real current-V2 video and audio frontend lifecycles pass. The retained
      2026-09-02 frontend proofs use Wan 2.2 TI2V 5B for video and MiniMax
      Music 3 for audio. Each proof starts from the registered V2 catalog
      Block, edits workflow-local controls, saves and reloads the workflow,
      verifies collapsed/expanded execution parity, submits the concrete graph
      through the visible frontend, waits for the exact backend task, and
      retrieves the attributed MP4 or WAV. Their bounded proof values do not
      replace the creator defaults, and neither proof grants public showcase,
      executable, Auto, or Gallery authority.
- [x] The combined client/backend V2 contract gate, full backend suite,
      migration UI gate, all 90 exact route audits, and focused registered-route
      promotion/runtime tests pass.
- [x] The final post-change full client check, complete mocked browser suite,
      clean-browser restart, and Git/evidence audit pass as one engineering
      regression gate. The latest Qwen-completion checkpoint on 2026-09-04 is
      126/126 mocked Studio browser tests, 2/2 shared-control browser tests,
      2,452 backend tests (54 optional-runtime skips), and a green production
      client build/bundle gate. The Qwen regression also verifies every
      retained admission-output byte or layered collection hash against its
      receipt. A stale mocked compiler helper was found and corrected during
      this gate: fixture pinning now applies the same reviewed initial-handle
      visibility stabilization and graph reconciliation as the production
      compiler, so tests cannot silently pin a pre-stabilized
      `BlockDefinitionV2`. The final move-out browser gesture now releases at
      a visible canvas point outside the Block instead of beyond the viewport.
      The supervised AMD backend is ready and idle. This does not override
      broader definition-specific video/audio, historical migration, resource
      qualification, or publication-evidence gates below.
- [x] The read-only legacy audit is complete: 375 instances span 102 immutable
      manifest identities, including 21 exact-current instances across two
      identities. The current Qwen route reseal made two previously current
      saved instances historical; they require separate reviewed recovery
      authority rather than silent identity rewriting.
- [ ] The remaining 339 blocked instances across 97 historical identities need
      reviewed compiler mappings or backend-pinned semantic-equivalence
      receipts; no cleanup or promotion may treat a current route, matching
      label, or model ID as migration authority. The 15 mapping-ready instances
      still require exact per-instance compiler output, read-only preview,
      explicit Apply, byte backup, and rollback before any workflow changes.
      Exhaustive local recovery found seven complete historical bodies only in
      byte-pinned retained live-node-library transcript records. Six exact
      manifest/admission/Studio tuples cover 42 instances and are registered in
      the read-only audit; the HunyuanVideo 1.5 body has no historical execution
      admission and remains explicitly non-convertible. No other historical
      hash occurs in Git history, the exact 2026-08-24 response, gzip frontend
      captures, reports, browser traces/caches, or retained uncompressed JSON
      artifacts. The migration journal/backup store is absent and the checked-
      in equivalence ledger is empty. The remaining 312 instances across 94
      identities need an authenticated old export/build artifact or an
      independently reviewed exact graph/interface equivalence receipt.
- [x] The checked-in publication-evidence audit now treats the former Qwen
      Image and MiniMax Music 3 schema-v1 approvals as historical and
      non-authorizing. The active schema-v2 promotion receipt ledger is empty,
      so both exact routes require explicit workspace-owner reapproval against
      their current admission, Studio-spec, artifact, manifest, and
      `BlockDefinitionV2` identities; Wan has no historical approval to reuse.
      No route currently receives `liveProof`, public executable, Auto,
      Gallery, or release-eligibility authority from the promotion ledger.
- [x] Three independent exact-route resource receipts now bind Qwen Image
      text-to-image, Wan 2.2 TI2V 5B text-to-video, and MiniMax Music 3
      text-to-audio to two distinct current-V2 frontend proofs apiece. Every
      receipt explicitly sets `familyCoverageDeclared`, `publicationAuthority`,
      and `autoAuthority` to false. They populate only the exact
      `routeQualifications` lane; the separate family recipe report remains
      12/51 and must not infer cross-template or cross-workflow coverage from
      these measured workloads.
- [x] The first generic registered text-to-image Block now offers Qwen Image
      2512, FLUX.1 Dev, and Stable Diffusion XL 1.0 as explicit exact routes.
      Each route remains one immutable reviewed `BlockDefinitionV2`; switching
      stores an independent inactive draft instead of mutating a loader or
      silently changing another Block. The retained backend workflow
      `1ZEaXrQ2Hl6kNBFXKavZB` was migrated through the deployed frontend to the
      superseded coarse Qwen pin `block-definition-v2-fa9fae25`, with its Qwen prompt and
      both FLUX and SDXL drafts preserved across Save.
- [x] Stale inactive registered drafts now rebase onto the current exact
      destination definition when—and only when—they contain no structural
      edits. All still-compatible controls and surviving internal layout are
      retained. A stale structurally modified draft fails closed with guidance
      to save it as a User Node, preventing either silent schema drift or loss
      of user structure.
- [x] Public connector projection is cached by immutable Block-instance
      identity. The graph fixer therefore no longer repeatedly normalizes the
      same registered Block while resolving every connector. Focused tests
      prove reuse for an unchanged instance and invalidation for copy-on-write
      updates.
- [x] Optimize generic route switching with multiple inactive drafts. An exact
      current inactive draft is restored by immutable ID/hash/SHA validation
      instead of recompiling the same hidden Modular graph. Stale or
      structurally incompatible drafts still fail closed or use the reviewed
      rebase path.
- [x] The development-proxy conditional-schema race is removed from the formal
      three-route Playwright proof. Qwen's reviewed route now deterministically
      replays the ordinary HandleField initial visibility result inside the
      isolated compiler, independent of whether React happened to mount the
      hidden transient connector first. The all-routes audit and the live Vite
      lifecycle both pass; the latter inserts a generic Block, edits Qwen,
      switches through FLUX and SDXL with undo/redo and sibling isolation, then
      saves, refreshes, and verifies all three independent drafts in 44.5
      seconds. The compiler's non-sensitive received hash/SHA diagnostic is
      retained for future route qualification.
