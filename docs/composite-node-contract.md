# Composite node contract

This reference preserves the normative Block V2 contract from the retired
implementation plan. It specifies required semantics, not blanket live-model
qualification. Historical milestones, test counts and rollout estimates are in
Git history; current support must be established from exact-route evidence.

The executable contract lives in `modiff/block_definition_v2.py`,
`modiff/composite_migration.py`, and the client’s `blockSchemaV2.ts`,
`blockRuntimeV2.ts`, `registeredBlockAdapterV2.ts` and persistence modules.
Keep client/backend canonicalization, validators and shared fixtures aligned.
See the [API reference](api-reference.md) for HTTP persistence/migration and
[workbench acceptance](workbench-acceptance.md) for qualification boundaries.

Normative **MUST**, **MUST NOT**, **SHOULD**, and **MAY** express requirements.

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
  containerInterface?: BlockContainerInterfaceV1;
  parentNodeId?: string;
};

type BlockGraphEdgeV2 = {
  edgeId: string;
  sourceNodeId: string;
  sourcePortId: string;
  targetNodeId: string;
  targetPortId: string;
};

type BlockContainerInterfaceV1 = {
  schemaVersion: 1;
  boundary: BlockBoundaryV2; // mode MUST be explicit
  controls: Omit<BlockControlV2, "defaultValue">[];
  previews?: BlockPreviewBindingV2[];
};
```

`containerInterface` is an optional, canonical/hash-covered local public
surface on an existing semantic node, not a nested composite instance. Its
ports and controls bind only to that node and its semantic descendants.
Primary/mirror bindings must resolve to existing direction- and
type-compatible fields; duplicate input targets, duplicate control targets,
and a shared input/control ID with different complete target sets are rejected.
Every current edge crossing a declared container boundary must have a matching
local port. Declaration edits that remove or rebind a connected port fail
before any graph mutation. Sealed controls cannot be weakened by this editor.

Root/local type compatibility has the same narrow file-transport exception:
an explicitly matching media file browser can bind a media port and a string
path control with the same full target set. Video file browsers also admit
declared sequences of PIL/image frames. Preview widgets qualify URL/base64
transport by their exact media display, rather than pretending URLs are image
tensors. Other string fields and mismatched modalities remain rejected.

There is still one flat effective execution graph and one owning instance.
Local controls MUST NOT store defaults or values independently. They write
existing effective fields or the root logical override when that field is
already exposed; shared root mirror consumers retain one value identity.
Renaming/reordering a local interface does not rewrite the root interface or
any field values. Newly wired non-baseline crossings retain a durable socket
after disconnection; neither projection nor persistence recreates a wire.
Collapsed mirrored sockets group only identical visible wires and carry exact
semantic-edge receipts for atomic connect/reconnect/delete. Execution always
uses the original flat graph, not those grouped canvas edges.
An undeclared deeper view inherits the nearest ancestor-local controls before
root controls, without adding a declaration during a value edit. A mirrored
public input remains one socket with its complete subtree-local target set.

Subtree adoption validates the complete incoming graph atomically and remaps
local bindings with semantic IDs. Replacement preserves compatible local IDs
and rebinds their targets; deletion reports declarations held by surviving
containers rather than silently dropping them. Saving a configured subtree as
a User Node promotes that exact local surface to the new root boundary and
controls, baking current field values/defaults into the copy. It does not
modify the source workflow or register a new catalog Cluster.

Local `previews`, when declared, bind existing compatible preview/output fields
within that same subtree. They share one owner `previewStates` inventory rather
than copying run state per view. Inventory order is root-definition bindings,
then first-seen local bindings in semantic graph-node order, deduplicated by
node/output port. Shared sources MUST agree on media type; a local `primary`
belongs only to that view, not to a newly introduced inventory entry. At most
one primary is allowed per surface. Root and local views filter this inventory
using their own declarations. The paired fixture is
`tests/fixtures/block_container_previews_v1.json`.

Nesting a whole multi-root Block introduces one generic non-executing group in
the target's flat graph. Its local interface is the source's public boundary,
controls and previews. All semantic IDs, edges, parent relations, exact upstream
placements and declarations are rebased together; effective values are baked
into copied fields. A single container with a different independently declared
surface also receives a wrapper, preserving both interfaces. The inserted outer
container begins collapsed. Moving within a workflow preserves completed media
references but never transfers in-flight execution authority. Saving it as a
new reusable definition preserves current prompts/settings and preview bindings,
not transient output media or publication approval. Generic grouping imposes no
pipeline family; actual upstream control-owner adoption remains pinned-family
checked. Typed execution connections remain authoritative in either case.

Compatibility: omitted declarations preserve old canonical bytes and hashes.
Legacy interfaces are derived until an explicit interface or non-baseline
wiring edit requires persistence. Readers without this optional-field contract
reject it under strict validation; they must not discard it. This requires
coordinated frontend/backend deployment, not an automatic historical-definition
rewrite. The cross-runtime fixture is
`tests/fixtures/block_container_interface_v1.json`; both strict validators,
hash implementations and API/workflow round trips are release gates.

`parentNodeId` is optional, canonical/hash-covered semantic ownership for an
ordinary node or a saved User subtree placed inside another semantic container.
It references a group or an exact non-leaf upstream block in the same graph.
Missing parents, self-parenting, cycles, leaf parents, and conflicts with an
already resolved upstream placement parent are rejected. Existing upstream
placements keep their exact provenance; ordinary utility nodes MUST NOT acquire
fabricated `modularDiffusers` identity merely because they are nested. Both
explicit ownership and upstream placement resolve through one parent relation
for projection, descendant scopes, validation and subtree persistence.

Moving/copying a subtree remaps internal parent IDs and local interface bindings;
saving it independently removes only its external parent reference. Current
field values and root overrides are baked into the reusable copy. Ownership
does not add an executor, second instance, new parameter value authority or
implicit wire. Palette insertion/adoption is atomic: failure removes the draft
insertion, and one Undo restores the prior graph. A whole Block with multiple
independent roots uses a generic wrapper carrying its already explicit public
surface. It never guesses a boundary or silently discards crossing wires.
Ordinary nodes and User containers can change parent within an instance while
preserving IDs, fields, wires and execution order. A retained local declaration
that would cross outside its owner must be rebound explicitly before the move.
Empty generic groups remain expandable insertion targets; inactive empty
upstream branch annotations do not become executable containers.
Whole-subtree move-out creates a workflow-owned User Block, not a library entry.
It preserves local controls and completed preview references; crossing wires
require exact public boundaries and complete fan-out. Existing root-owned
controls and previews are protected until explicitly rebound. The shared fixture is
`tests/fixtures/block_parent_node_v2.json`. Omission preserves pre-extension
hashes; older strict readers reject the new field, requiring paired deployment.

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

## Migration and recovery

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
