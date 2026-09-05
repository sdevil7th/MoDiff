# Local API Reference

MoDiff serves its bundled client and backend API from the same origin, normally `http://127.0.0.1:8088`.

This API is unversioned, unauthenticated, and intended for the trusted local MoDiff client. It is not a public multi-user API. Several routes mutate files, download models, import Python code, or execute graphs; read [SECURITY.md](../SECURITY.md) before writing another client or changing the bind address.

Every route enforces the local request boundary before its handler runs,
including read-only workflow, queue, and media metadata. The request Host and
connected peer must be literal loopback addresses (or `localhost`). A browser
request that supplies an Origin must use a loopback HTTP(S) Origin; the Vite
development origin on `localhost` is supported even when the backend URL uses
`127.0.0.1`. Originless native clients and ordinary browser navigations are
accepted only over loopback. Do not use an arbitrary DNS name that happens to
resolve to loopback.

## Route groups

| Area                    | Routes                                                                                                                                                                                                                                                                                               | Purpose                                                                                                                                                                                   |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Bundled client          | `GET /`, `/favicon.ico`, `/assets/*`, optional `/template-gallery/*`, `/user/*`, `/static/{module}/{file}`                                                                                                                                                                                           | Serve the generated frontend and module UI assets. The Gallery route exists only for an explicit offline/local asset build; normal releases use an immutable public Hugging Face Dataset. |
| WebSocket               | `GET /ws`                                                                                                                                                                                                                                                                                            | Session handshake, queue restoration, progress/events, field signals, and node updates.                                                                                                   |
| Registry                | `GET /nodes`                                                                                                                                                                                                                                                                                         | Return the live registered node contracts used by the bundled client.                                                                                                                     |
| Execution               | `POST /graph`, `GET /queue`, `GET /runs/{task_id}`, `DELETE /queue/{task_id}`, `POST /stop`                                                                                                                                                                                                          | Queue, inspect, remove, or interrupt graph work. A normally supervised backend replaces its worker when a blocking model call misses the cancellation grace period.                       |
| Node state              | `POST /fields/action`, `GET /cache/{node}/{field}[/{index}]`, `DELETE /cache`                                                                                                                                                                                                                        | Run declared dynamic field actions and access/clear statically declared media or text cache fields. Connector, process-local, and other opaque outputs are not cache-servable.            |
| Files and graphs        | `GET /listdir`, `GET /listgraphs`, `GET /file`, `POST /file`, `GET /preview`, `GET /stream`                                                                                                                                                                                                          | Browse the configured working directory, load/save graph files, upload media, and stream previews.                                                                                        |
| Saved workflows         | `GET /workflows`, `GET/PUT/DELETE /workflows/{workflow_id}`                                                                                                                                                                                                                                          | List, read, replace, or delete versioned workflow records below the configured data directory.                                                                                            |
| Media I/O               | `GET /media/capabilities`, `/media/probe`, `/media/export`, `/media/preview`                                                                                                                                                                                                                         | Inspect a managed media identifier or return a cached, converted download/browser preview through the built-in deterministic media tools.                                                 |
| Runtime                 | `GET /health`, `/runtime/status`, `/runtime/resources`, `/runtime/options`, `/system_stats`, `/runtime/gpu_processes`; `POST /runtime/gpu_cleanup`                                                                                                                                                   | Read readiness, resource, option, and hardware state or request best-effort runtime cleanup.                                                                                              |
| Optimizations           | `GET /runtime/optimizations`, `/jobs/{job_id}`, `/receipts`; `POST /runtime/optimizations/install`, `/activate`, `/rollback`, `/enable`, `/probe`, `/qualify`, `/jobs/{job_id}/cancel`                                                                                                               | Inspect runtime features and legacy package contracts, manage recovery, and record bounded local qualification evidence. Hashless package install and activation are unavailable.         |
| Optional model runtimes | `GET /runtime/optional-runtimes`, `/jobs/{job_id}`; `POST /runtime/optional-runtimes/install`, `/activate`, `/rollback`, `/jobs/{job_id}/cancel`                                                                                                                                                     | Publish the reviewed optional-library contract and its fail-closed staged lifecycle. The current candidate exposes no executable install or activation action.                            |
| Auto resource           | `POST /auto_resource/plan`, `POST /auto_resource/plans`, `GET /auto_resource/history`, `DELETE /auto_resource/history`                                                                                                                                                                               | Plan hardware-aware model recipes and manage local planner history.                                                                                                                       |
| Models                  | `GET /huggingface/node-library`, `/huggingface/modular-conditionals`, `/huggingface/registered-block-v2`, `/model_capabilities`, `/model_artifact_catalog`, `/model_fingerprints`, `/local_models`, `/hf_cache`, `/model_cache/diagnostics`, `/hf_hub`, `/hf_download/plan`; `POST /hf_download`, `/hf_token`; `DELETE /hf_cache/{hash}` | Discover reviewed first-party node definitions, exact compiled Block definitions, and unpruned Modular branch contracts; diagnose, space-plan, download, authenticate, fingerprint, and delete model artifacts.              |
| Template Gallery setup  | `GET /template_gallery/status`, `/template_gallery/plan`; `POST /template_gallery/install`                                                                                                                                                                                                           | Inspect, space-plan, and explicitly install or repair the byte-pinned Gallery payload through the local app.                                                                              |
| Media lifecycle         | `GET /media_assets`, `DELETE /media_assets`                                                                                                                                                                                                                                                          | Inspect temporary media records or remove exact unpinned, task-scoped, or age-scoped files while no generation is active.                                                                 |
| Custom modules          | `GET /custom_modules`; `POST /custom_modules/refresh`, `/install`, `/{name}/update`, `/{name}/disable`, `/{name}/enable`                                                                                                                                                                             | Clone/copy and import trusted custom Python modules or change their enabled state.                                                                                                        |
| Studio outputs          | `GET/POST /studio_outputs`, `PATCH/DELETE /studio_outputs/{output_id}`                                                                                                                                                                                                                               | Persist and manage local Studio output metadata and copied media.                                                                                                                         |
| Studio blocks           | `GET/POST /studio/blocks`, `GET/DELETE /studio/blocks/{block_id}`                                                                                                                                                                                                                                    | Persist reusable local graph blocks.                                                                                                                                                      |
| Composite migration     | `GET/POST /studio/composite-migrations/preview`, `GET /studio/composite-migrations`, `/recovery-audit`, `/{migration_id}`; `POST /studio/composite-migrations/apply`, `/{migration_id}/rollback`                                                                                                 | Inspect redacted partial recovery evidence or explicitly apply safe V1/exact compiler-supplemented Cluster conversions with exact local backups and fail-closed recovery.                  |
| Workflow shares         | `GET /workflow_shares`, `POST /workflows/share`, `GET /workflows/share/{share_id}`, `GET /workflows/share/{share_id}/media/{filename}`                                                                                                                                                               | Create and render local workflow share packages and their copied preview media.                                                                                                           |

## Core response contracts

### Managed file identifiers

`POST /file` stores uploads below the configured data directory and returns a
portable `@data/<relative-path>` identifier, such as
`@data/images/source.png`. Pass that opaque identifier unchanged to graph
inputs and to `/file`, `/preview`, `/stream`, and `/media/*` routes. It is
data-root-relative, contains no absolute host path, and works when `data_dir`
is outside `work_dir` or on another Windows drive. The backend rejects `..`,
malformed namespace values, and symlink escapes. Existing work-root-relative
values such as `data/images/legacy.png` remain readable when they resolve
inside a configured root.

### Runtime status

`GET /health` and `GET /runtime/status` use the same readiness handler. The response includes:

- `ready` and `error` summary flags.
- Backend `instance` identity.
- Immutable `backend_source` process-start identity. Its fingerprint is also
  carried as `runtimeFingerprint.backendSource` on execution-completion
  receipts so qualification tooling can reject a stale worker after source
  files change.
- Python and required-package status.
- Registered module counts.
- Server and relevant configuration state.
- Queue summary.
- Normalized `hardware` data.

`GET /system_stats` returns the normalized hardware snapshot directly. The current schema includes `schema_version`, `system`, `torch`, `devices`, `default_device`, and `disk`. Callers should tolerate additive fields and individual probe errors.

`GET /runtime/resources` returns a short-lived schema-versioned sample of the
MoDiff process, storage roots, detected accelerators, active device, and current
run. Accelerator shared/dedicated memory classification follows the hardware
probe when it identifies the topology; capacity-ratio heuristics are only a
fallback for unknown topology. A large dedicated accelerator must not be
reported as shared memory merely because its VRAM approaches host RAM capacity.
Storage capacity remains available as `usedBytes`, `freeBytes`,
`totalBytes`, and `percent`; `activePercent` is interval I/O active time for the
disk backing the MoDiff data path. On Windows it is sampled from the physical
disk idle/query counters used by the operating-system performance telemetry.
Unsupported or inaccessible activity counters return `null` rather than
substituting capacity used. `GET /runtime/options` returns the live node option descriptors after
device and package compatibility filtering. Both are observations, not proof
that a real model workload completed.

### Contract-only Diffusers capabilities

`GET /huggingface/node-library` returns the schema-v6 immutable first-party
Diffusers Cluster Node catalog derived from the pinned no-weight Modular
workflow snapshot. A Cluster Node contains one reviewed upstream
pipeline/workflow definition, but is not named a pipeline in the node UI. Each
definition identifies its exact pipeline class, blocks class, workflow,
hierarchy, inputs, outputs, components, integration status, and content hash.
Schema v6 publishes a deduplicated `blockDefinitions` collection with the
exact declared block inputs, variadic `kwargs_type` inputs, outputs, required
inputs, per-block components, configs, and content hashes. Each Cluster
definition references its aggregate workflow contract through
`rootBlockDefinitionId` and its visible child contracts through ordered
`blockPlacements` whose
`path` is an exact segment array. The array is authoritative: upstream permits
a top-level block name to contain a dot, so `legacyPath` is display and
compatibility metadata and must not be split to infer nesting.
`GET /huggingface/modular-conditionals` is the separate schema-v1 companion for
the original, unpruned `pipeline.blocks` trees. It preserves all conditional
containers, branch alternatives, trigger-input order, defaults, complete
presence/absence selection tables, rejected input combinations, and the
selected branch/active leaf trace for each advertised workflow predicate. The
snapshot covers the same 34 pipeline classes and 94 workflows without changing
their existing resolved definition identities. Clients use this contract only
when they need to display or edit upstream alternatives; normal Cluster
materialization may continue to use the smaller resolved node-library graph.
Schema v6 also publishes 321 deduplicated `blockRoleAdapters` for all 50
contract-only workflows and the five definitions that previously had only an
equivalent standard route. They map exact content-addressed block definitions
to reusable semantic roles (including the workflow root) and retain
`executionClaim: structural_adapter_only`; they do not select a model artifact
or claim that the block is executable. Nine exact `containerStateAdapters`
record the loop-local initialization, iteration cardinality, published state,
and progress semantics required by the 12 Helios, MiniMax Music, and Wan
Animate custom-container workflows.
For exact workflows already represented by MoDiff's reviewed generic action
truth, `graphAdapterContracts` also declares the action-role sequence and state
edges. These contracts remain discovery metadata; an entry still requires a
separate materializer, execution profile, artifact qualification, and runtime
evidence before it can run or enter Auto.
Schema v6 also attaches schema-v4 `executionAdmissions` to the explicitly
reviewed publication candidates. Each admission binds one exact Cluster
definition, graph-adapter contract, Studio execution-spec hash/profile, and
immutable artifact revision. It also declares the complete Studio binding-source
set, the exact mapping from upstream Cluster inputs to those sources, and the
remaining sources that a Cluster instance may persist as execution-parameter
overrides. Artifact, pipeline-class, repository, revision, and fixed literal
sources are sealed and cannot be instance overrides. Candidate admissions
additionally declare the ordered dynamic field actions
(`onChange`/`onSignal`) and binding value source needed to ask the existing
generic Modular Diffusers nodes for their exact backend-owned field schemas.
The admission carries backend-sealed values and exact auxiliary model
dependencies, so a client cannot replace the base artifact, pipeline class, or
Qwen ControlNet repository/revision through generic binding values.
The generated registered-Block catalog currently contains 90 admitted static
joins: 82 Diffusers joins and eight Transformers composites. The Diffusers
joins include official Modular Diffusers workflow definitions and reviewed
Studio execution composites; all eight Transformers joins are reviewed Studio
execution composites represented by the same source-neutral V2 contract.
Every admission still has `executable: false`: admission and V2 insertion prove
the graph contract, while installed artifacts, runtime/resource readiness, and
an actual qualification run remain separate evidence.
The reviewed pin currently contains 94 Modular Diffusers Cluster definitions,
additional Diffusers Studio execution composites, and eight Transformers
Studio execution composites. Its unpruned Modular registry contains 632 exact
definitions: 598 placed upstream definitions and 34 pipeline roots. Definitions
also carry their exact graph-adapter contracts; consumers must read those
contracts from the generated catalog instead of depending on a duplicated
documentation count.
Definitions from different model families implementing the same task reference
one shared task-contract ID, such as
`diffusers.task.text_to_image.v1` or `diffusers.task.image_to_video.v1`.

Every schema-v6 definition has `executionClaim: discovery_only`. The catalog is
structural metadata, not evidence that a model is installed, runnable,
qualified, or eligible for Auto. Reading it does not import Diffusers,
Transformers, or Torch and does not load or download model weights. Built-in
definitions have `surface: diffusers_cluster_nodes`, `ownership: library`, and
`mutable: false`; custom blocks remain User Nodes under `/studio/blocks`.

`GET /huggingface/registered-block-v2?definition_id=...&admission_id=...`
returns one build-time generated `BlockDefinitionV2`, its initial instance
values, recursive internal layout, and canonical SHA-256 for the exact
schema-v6 definition/admission pair. The backend revalidates all definition,
manifest, admission, graph, interface, and canonical hashes before serving a
detached copy. The browser repeats the source identity and canonical SHA-256
checks before graph mutation. This request never imports a model runtime,
executes a node field action, accesses the Hub, or downloads weights.

Catalog visibility is not execution authority. An admitted definition is shown
as **Graph qualified** and made clickable/draggable only when its exact
registered route and compiled catalog entry agree. Reviewed Modular Diffusers
definitions without an executable admission may still be inserted as
source-neutral structural V2 Blocks with no Run or Auto authority. No fresh
insertion creates a legacy `cluster` renderer or invokes the legacy hidden
dynamic-field compiler. That compiler exists only at the explicit V1 recovery
boundary, where the archived/equivalence receipt rules still apply.

The generated catalog is maintained from the authoritative backend registry
and current route pins. From `MoDiff-client`, run
`npm run catalog:block-v2:generate` after any node schema, definition,
admission, route, or Modular hierarchy change. `npm run
catalog:block-v2:check` independently recompiles all routes and byte-compares
the checked-in compressed catalog; it is part of both primary validation
gates. The generator uses a private temporary directory and removes it on
success or failure.

Publication flags remain independent of insertion. The exact checked-in
evidence audit and blocker semantics are documented in
`docs/huggingface-cluster-publication-evidence.md`; run
`scripts/audit_huggingface_cluster_publication.py` to verify them without
loading a model or mutating a workflow. In particular, a route-specific output
approval grants only `liveProof`, and schema-v2 approvals must also bind the
current registered `BlockDefinitionV2` content hash and canonical SHA-256.
Legacy schema-v1 approvals are preserved as non-authorizing history and require
fresh owner review after a Block identity change. Family-level resource
coverage cannot grant exact-route Auto authority. New approvals can be staged
with `scripts/stage_huggingface_cluster_promotion_receipt.py`; the command
validates one
current frontend provenance document and its exact media bytes, requires an
explicit workspace-owner approval flag plus all seven lifecycle confirmations,
and emits only a deterministic, non-installing atomic candidate bundle. The
post-promotion manifest and Block pins come from the client route audit's
explicit `MODIFF_POST_PROMOTION_ADMISSION_ID` candidate mode, because changing
`liveProof` changes the manifest hash embedded in the Block source. The bundle
keeps the executed pre-promotion route and proposed post-promotion route
separate and includes any dependent historical compiler-mapping candidate. It
never edits the authority ledger or infers approval, time, comment, or
lifecycle evidence. The
separate
`routeQualifications` lane must bind
the admission, registered `BlockDefinitionV2` content hash and canonical
SHA-256, Studio execution specification, and immutable artifact revision; it
also binds every immutable admission dependency and the retained proof's full
executed model-set hash. Legacy reports without the lane normalize to an empty
unbound lane, and existing family measurements are never upgraded by
inference. It still does not grant Auto without a distinct publication authority. Local
review media cannot grant Gallery publication.

### Studio composite block persistence

`GET/POST /studio/blocks` and `GET/DELETE /studio/blocks/{block_id}` retain the
legacy `UserBlockDefinition` V1 contract during the compatibility window and
also accept the canonical user-owned `BlockDefinitionV2` contract. V2 stores
the exact graph, explicit or one-time-derived boundary, ordered controls,
creator suggestions, preview bindings, provenance, ownership, and canonical
graph/definition hashes. The backend validates and round-trips those fields; it
does not infer new ports, replace instance values with defaults, or silently
rewrite content hashes. Public `portId` values must be unique across both
boundary inputs and outputs because the shared connector map and React Flow
root-handle namespace use one ID space; client and backend validators reject a
cross-direction collision.

One logical input or control may explicitly fan out to several internal
fields through `mirrorBindings`. The primary binding remains required;
mirrors are canonical, unique, type-compatible targets and are included in
definition/interface hashes. Mirrors are supported for public inputs and body
controls only—public outputs remain single-binding. The client projects one
saved value atomically to every declared target, shows one public socket or
control, expands its root bridge to the complete target set, and expands one
external input edge into deterministic execution edges. Registered routes
must declare the exact fan-out and prove that initial/default values agree;
neither the client nor backend infers it from matching names or values.

The route is a User Node store. A definition whose source is the registered
Diffusers or Transformers catalog, or whose ownership is registered, is
rejected. Registered definitions remain read-only catalog data even though
they share the same V2 definition schema and are required to converge on the
same canvas behavior. Hub imports must use an exact immutable repository
revision. Nested composite definitions are rejected for both V1 and V2.
`BlockInstanceV2` is not stored by this route. Under the V2 workflow contract,
the workflow record embeds each instance snapshot, values, effective graph,
effective interface, layout, preview state, and authorities so one insertion
cannot reset or mutate another. The effective interface contains the ordered
boundary/controls plus definition-base and effective hashes; older V2 records
without it deterministically migrate to the embedded definition interface.
Workflow normalization now canonicalizes any already-V2 root from that
embedded instance and removes derived V2 projection children and edges. The
exact live route table currently sends all 90 registered admissions through
this V2 representation: 82 Diffusers and eight Transformers, including all
three admitted speech routes. No currently graph-qualified admission falls
back to the legacy Cluster renderer; each route remains independently pinned
to its reviewed definition, immutable artifact, Studio execution receipt,
compiled definition content hash, and canonical-definition SHA-256.

The client exposes three explicit V2 persistence choices. **Keep only in this
workflow** writes no reusable definition. **Save as new User Node** posts a new
user-owned definition containing the instance's exact effective graph,
existing explicit interface, suggested inputs and preview bindings, direct
parent provenance, and current values as defaults wherever a declared control
can represent them. **Update existing User Node** uses the same preservation
contract but is available only for a mutable user-owned source and can never
overwrite registered catalog data. Since `BlockPortV2` has no `defaultValue`,
either reusable-definition operation fails closed if an instance value belongs
to a public boundary input that is not also a declared control; the value may
still remain embedded in that workflow instance.

V2 definition saves are optimistic in the Node Library and roll back on an API
failure. After an acknowledged response, the client verifies the exact
normalized definition and checks that the initiating instance did not change
semantically while the request was pending. Only that target instance is then
rebased; sibling instances and snapshots in other open workflows retain their
embedded definitions. Saved V2 User Nodes appear alongside V1 User Nodes and
can be clicked or dragged into a graph, producing a new independent top-level
`BlockInstanceV2`. Focused store/runtime tests cover rollback, semantic races,
sibling/open-workflow isolation, and reinsertion. A mocked browser lifecycle
also exercises all three visible save choices, node adoption, workflow
Save/refresh, and the rebased updated instance. The complete release browser
suite remains a separate gate rather than an unimplemented persistence feature.

The synchronized executable contract lives in
`MoDiff-client/src/studio/blockSchemaV2.ts` and
`MoDiff/modiff/block_definition_v2.py`; `/studio/blocks` dispatch is in
`MoDiff/modiff/server.py`. Both canonicalizers must produce the same exact
`block-graph-v2-*` and `block-definition-v2-*` hashes. The registered catalog
compiler is `MoDiff-client/src/studio/registeredBlockAdapterV2.ts`; it converts
one exact graph-qualified Diffusers or Transformers admission and complete
materialized skeleton into the common definition/instance contract while
keeping mutable values, internal layout, generated previews, progress, and UI
state out of reusable-definition hashes. Legacy `huggingFaceCluster*` binding
and receipt options are compiler inputs only and are removed after their
meaning is represented by explicit V2 controls, boundary, values, and source
metadata. Exact Modular Diffusers semantic nodes may carry the optional
hash-covered `modularDiffusers` object documented by the normative composite
contract. It identifies infrastructure versus a pinned upstream block and
records the exact pipeline/workflow/block placement needed for deterministic
runtime lowering; it does not store parameter values or layout. Catalog-copied
upstream nodes retain `sourceDefinitionId`, `sourcePlacementPath`, and
`sourceExecutionScope` together. The editable `placementPath` records where
that instance now lives; the immutable source triplet records what reviewed
block/subtree was copied. Insert/replace composition recipes are therefore
derived without class-name or path guessing. The backend applies those recipes
to the pinned unpruned `pipeline.blocks` tree, resolves the requested workflow,
and only then invokes upstream `init_pipeline()`. A structural rebuild receipt
is not a model-execution or publication claim. Upstream workflow IDs,
placement-path segments, component names, and graph-field bindings preserve
their exact Python identifiers and may begin with `_` (including the
`__unpruned__` catalog sentinel and fields such as `_auto_resize`). Public
node, port, control, and suggestion IDs retain the stricter public-ID grammar.
A V2 field or
hash change must update all affected client/backend
types, validators, canonicalizers, adapters, fixtures, this API reference, and
the normative contract in one change. When it affects V1 reading, the client
`migrateUserBlockDefinitionV1` and backend
`migrate_user_block_definition_v1` adapters must also change together and
produce the same graph/definition hashes. Run
`MoDiff/scripts/verify-block-v2-contract.sh` as the combined client/backend
contract gate, including reusable-definition persistence and V1 migration/
recovery; set
`MODIFF_CLIENT_ROOT` only when the client checkout is not the backend's sibling
directory.

`MoDiff-client/src/studio/compositeBlockCapabilitiesV2.ts` computes transient
instance/save actions and is not persisted as definition data or execution
authority. Structural editing and Configure Interface are enabled only for a
valid definition/instance pair and the relevant workflow/workspace permission.
The shared connector path derives public sockets from the instance
`effectiveInterface` and commits admitted same-owner internal edge additions,
removals, and reconnections to `effectiveGraph` copy-on-write. Eligible
deletion, disconnected adoption, adoption of an edge already routed through a
declared root port, compatible node replacement, and move-out through an
existing public port are atomic, undoable, authority-invalidating semantic
edits of the same root. An exact Modular Diffusers container dragged from the
left catalog first materializes as a V2 fragment; when dropped into an
expanded compatible Block, its complete ordinary subtree is copied, rebased,
and flattened into that root's effective graph. No nested Cluster/User node is
persisted, and its immutable source definition/path/scope remains hash-bound.
Nested composites, incompatible bindings, arbitrary
external crossings, and removal of a connected public port fail visibly before
mutation. A compatible replacement for a preview or sealed-control owner uses
an identity-preserving path: the protected semantic node ID and immutable
binding remain unchanged, every referenced field must retain its type and
direction, previews must retain their media/display role, and stale preview
media/task state is cleared. Arbitrary crossing uses the explicit flow
disconnect → move → Configure Interface → reconnect; the editor never invents
a public boundary from topology.

Configure Interface edits ordered inputs, outputs, and body controls in the
workflow instance. Binding candidates come from the effective internal graph;
the validator checks node/field existence, direction, value type, stable port
identity, shared input/control binding, control order, interface hashes, and
sealed controls. An edge-impact preview disables Apply when a connected port
would be removed. Save-as-new/update promotes the exact effective graph and
effective interface into the user-owned reusable definition; registered source
definitions remain immutable.

The same editor reorders entries and manages the additive `mirrorBindings`
contract for public inputs and editable controls. It offers only compatible,
non-duplicate internal consumers, writes mirrors in canonical order, does not
offer output fan-out, and keeps sealed-control mirrors read-only.

Interface relabel/reorder preserves existing mirror bindings when the primary
binding is unchanged. Explicit primary rebinding clears them until the user
declares the new complete fan-out. Replacement, deletion, sealed-control
checks, execution export, and interface hashing inspect the primary and every
mirror together; a partial or divergent fan-out fails closed.

The implemented pure V2 canvas projector/controller is
`MoDiff-client/src/studio/blockRuntimeV2.ts`. Its React Flow roots, controls,
connector views, child nodes, bridge edges, copy-on-write reducers, and
execution expansion are derived from one embedded `BlockInstanceV2`; they are
not additional persistence contracts. `normalizePersistedFlowState` in
`MoDiff-client/src/stores/useFlowStore.ts` invokes its persistence
canonicalizer for V2 roots. The runtime fails closed for malformed or stale
projections, missing declared bindings, unsupported node/handle/type semantics,
duplicate identities, and projection-ID collisions. Its current 24 focused
tests cover single-authority persistence, projection validation, mirrored
value/edge fan-out, root-to-root fan-out, and identical collapsed/expanded
execution expansion.

The schema, backend definition store, registered compiler, capability resolver,
pure runtime/projector, and V2 persistence canonicalizer are implemented.
Already-V2 roots also render through the `BlockNode` React Flow entry and one
shared `BlockNodeFrame`, with V2 controls, connector trays, resize behavior,
expansion, and refresh projection derived from the embedded instance. Public
connector lookup, connection type/color handling, pane drop, connection
mutation, connection status, and signal propagation also consume the explicit
V2 boundary instead of a second root-parameter schema. External connections
remain attached to the durable root; projection children are restricted to
same-instance internal edges.

The root's public input/output trays stay mounted when a Block is expanded.
Expansion does not retarget durable external edges to replaceable projection
children; it adds internal nodes and bridge links while the same stable root
handles remain available for surrounding-workflow composition.

API export and run readiness now expand already-V2 roots into the same concrete
internal graph for collapsed and expanded views. Run-target resolution maps a
root to its primary (or first declared) preview child, and malformed V2 graphs
produce a blocking `composite_execution_graph_invalid` issue instead of
leaking a wrapper into backend execution. Matching WebSocket progress resolves
back to the durable collapsed root, while a generated `update_value` for an
exact declared preview binding updates only that instance's preview media,
task, and status. It never writes an output into root `data.params`, a reusable
definition, sibling instances, or graph undo history. Registered catalog
insertion is now V2 for all 90 exact admissions in the live route table. Fresh
insertion fetches one precompiled, hash-pinned definition from
`/huggingface/registered-block-v2`; it does not create a hidden transient
compiler graph, run field actions, load a model, or access the Hub. The legacy
dynamic-field compiler is retained only behind the explicit V1 recovery
boundary. Catalog-only definitions are structurally insertable through the
same renderer but carry no execution or Auto authority, and a direct
route-mismatched executable insertion fails before creating a durable root.
Routed V2
instances use the instance-scoped Auto authority contract described below;
structurally or publicly customized instances remain independently runnable in
Expert/manual mode but fail closed in Auto. The catalog is not considered
broadly integrated until every exact admission passes that same compiler and
boundary gate. All three V2 persistence choices are implemented, including
registered-update rejection, rollback/race handling, target-only rebasing, and
click/drag reinsertion of saved V2 User Nodes. The validated instance-only
effective-interface contract and Configure Interface editor are now wired. A
mocked real-pointer browser lifecycle covers compatible replacement,
incompatible atomic rejection, public-edge adoption, explicit interface mirror
fan-out, internal/external reconnection, move-out through an existing public
port, undo/redo, Save/refresh, and re-expansion. Preview- or sealed-control-owner
replacement follows the stricter identity-preserving compatibility contract;
it never rebinds an immutable preview or sealed control to the dropped node's
identity.

A move-out gesture records the source Block before replacing its projected
child with an ordinary top-level node. For the remainder of that pointer
gesture, target discovery excludes only that source Block (including its
legacy User Node, V2 root, and projected-child views). This prevents the newly
materialized node from being immediately adopted back into the Block it just
left, while preserving direct move-out-and-drop into a different expanded
Block. The complete transition remains one undoable copy-on-write operation.

Expert/manual readiness preserves the issue's declared blocking policy. A
model, environment, hardware-fit, package, backend, or media finding
deliberately demoted to a non-blocking Expert warning must remain visible and
actionable in Graph Fix through its explicit **Models**, **Setup**, or
**Gallery** route. It must not disable **Run**, and Graph Fix must not silently
install, repair, or select anything. A malformed concrete V2 execution graph
remains a blocking `composite_execution_graph_invalid` error in Expert mode.

See the
[unified composite-node contract](unified-composite-node-implementation-plan-2026-09-01.md)
for the normative `BlockDefinitionV2`/`BlockInstanceV2`, hashing, copy-on-write,
migration, and schema-maintenance rules.

### Legacy composite inventory and Block V2 migration

The backend retains the deterministic, read-only operator inventory for the
JSON stores used by the workflow and User Node APIs:

```bash
./scripts/with-runtime-env.sh ./.venv/bin/python \
  scripts/inventory_legacy_composites.py --data-dir /path/to/data
```

It inspects only `user-workflows/*.json` and `studio/blocks/*.json` beneath the
selected data directory and prints JSON to stdout. `--compact` prints canonical
one-line JSON. `--fail-on-errors` prints the same complete report and exits 2
when any error-level issue is present. The CLI has no output-file option and
the inventory module itself remains read-only.

The current report contract is:

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

Each workflow report inventories legacy Cluster roots/derived children, V1
`userBlockSnapshot` roots/children, already-V2 roots/projections, and ambiguous
mixed-authority records. Composite entries retain IDs and source references,
prompt/parameter values, overrides and V2 instance values, canvas size and
position, persisted presentation, declared/projected ports and bindings,
derived-child IDs, external edge endpoints/handles, and reusable-definition
resolution. Reusable block entries classify V1 versus V2, include graph
node/edge IDs and counts, ports, prompts, parameters, source ancestry, and the
strict V2 validation outcome. Summary counters, sorted issue records, source
SHA-256 values, and `reportHash` support deterministic review.

The scanner rejects symlink sources, bounds file counts and byte sizes, parses
strict UTF-8 JSON without non-finite constants, and reports malformed or unsafe
records. It does not scan the registered catalog; those references are marked
external. A matched User Node record is informational because the embedded
workflow snapshot remains the later migration authority. Ambiguous/missing
references and malformed topology require human review.

Reports contain persisted prompt and parameter values. Treat stdout and any
operator-created redirection as sensitive plaintext. The report is not a
backup, executable readiness evidence, conversion authority, or permission to
delete or merge records. The inventory implementation remains
`modiff/composite_migration_inventory.py` with non-mutation fixtures in
`tests/test_composite_migration_inventory.py`.

`GET /studio/composite-migrations/preview` builds a separate deterministic
conversion plan from the current exact source bytes. It returns a
`migrationId`, `planHash`, inventory/source-set hashes, candidate dispositions,
and before/after hashes for each file that would change. It writes nothing and
does not return prompt or parameter values. The preview converts only the
parts whose meaning is complete:

- V1 reusable User Node definitions use the same additive conversion rules as
  `migrateUserBlockDefinitionV1` in the client. Definition IDs, materialized
  boundary ports/bindings, graph nodes and edges, control defaults, preview
  bindings, and origin provenance feed the synchronized V2 hashes.
- V1 workflow instances retain the root ID, current input/control values,
  root position and size, internal node IDs/layout/data, effective internal
  topology, and external edge IDs. An external edge attached to an expanded
  child is rebound to the same declared public root port. Missing or ambiguous
  bindings block that workflow conversion instead of dropping an edge. Every
  newly persisted instance includes a materialized `effectiveInterface` equal
  to its migrated definition boundary/controls and the exact synchronized
  `block-interface-v2-*` base/effective hashes; it does not depend on the
  client's additive legacy-read fallback.

`BlockInstanceV2` may additionally contain `routeSelection` schema v1 for a
generic registered model selector. The active `definitionSnapshot` remains one
exact immutable route. `inactiveDrafts` are bounded non-recursive snapshots of
other immutable registered definitions and their workflow-local values,
effective graph/interface, and internal layout; they cannot carry an instance
ID, preview/task state, or authority receipt. Client and backend validators
reject selected-route duplication, key drift, mutable/user definitions,
recursive fields, unknown fields, and invalid nested hashes.

The shared Block renderer offers three explicit outcomes when the active route
has workflow-local customization: cancel, keep the exact active route as an
inactive workflow draft and switch, or save the active effective
graph/interface/values as a new User Node and then switch. The save-and-switch
operation writes the reusable User Node first, verifies the acknowledged bytes
and workflow operation context, and leaves the canvas instance byte-identical
until that save succeeds. It never overwrites a registered definition. Route
switching then clears previews and execution/resource authority and compiles
the selected immutable route normally. Only the selected route expands into
the executable node map; inactive drafts may remain in the separately stored
workflow snapshot for reproducibility.

- Legacy registered Clusters are listed as blocked by the default GET. The
  backend will not infer an executable definition from presentation children.
  Each blocked candidate includes the backend-computed `sourceSha256` and
  `legacyCompositeHash` needed to bind a client compiler witness to those exact
  saved bytes; the default request still returns no prompt or parameter values.
  `POST /studio/composite-migrations/preview` accepts only
  `{"compilerSupplement": ...}` and can mark a Cluster convertible when its
  exact registered compiler output passes every source, identity, pin,
  topology, interface, value, layout, and external-edge receipt below.
  Already-V2 records are unchanged.

The compiler supplement is versioned independently:

```json
{
  "schemaVersion": 1,
  "kind": "registered_cluster_v2_compiler_supplement",
  "compilerOutputs": [
    {
      "sourcePath": "user-workflows/example.json",
      "sourceSha256": "sha256:...",
      "conversions": [
        {
          "legacyInstanceId": "cluster-root-id",
          "legacyCompositeHash": "sha256:...",
          "admissionId": "diffusers.cluster-admission:...",
          "compiledDefinitionContentHash": "block-definition-v2-...",
          "compiledDefinitionCanonicalSha256": "sha256:...",
          "blockInstanceV2": {},
          "ownedNodeMappings": [],
          "portMappings": [],
          "valueMappings": [],
          "previewMappings": [],
          "absorbedInternalEdgeIds": []
        }
      ]
    }
  ]
}
```

Outputs and conversions must be unique and canonically ordered. The source
SHA-256 and `legacyCompositeHash` bind the compiler output to exact reviewed
bytes. `blockInstanceV2` is strictly normalized by the backend, must keep the
legacy root ID/position/size/expanded state, must use an immutable registered
definition, must not contain a structural fork or carried execution authority,
and must match both entries in
`REGISTERED_BLOCK_V2_DEFINITION_PINS`: the public content hash and canonical
SHA-256. Its manifest definition/revision/hash and admission ID must also equal
the legacy Cluster receipt.

`ownedNodeMappings` maps every derived legacy projection node one-to-one to a
semantic graph node and receipts its exact internal layout. `portMappings`
binds every legacy public or externally connected endpoint to an explicit V2
boundary port by semantic node/field binding, never by label. `valueMappings`
must account exactly once for every durable persisted root/child parameter
value and both legacy override stores; each destination value must compare
equal. A legacy parameter declared with `display: "output"` is not a durable
input value and MUST NOT be copied into `values` or graph parameters. A
non-null output media reference instead requires one exact `previewMappings`
entry whose V2 preview binding has the same `mediaReference`; absent, duplicate,
or value-changing preview receipts block the conversion. Null output values do
not require a receipt.
`absorbedInternalEdgeIds` must exactly list the projection-only internal edges
that the embedded V2 graph replaces. All unrelated nodes and edges remain
unchanged; every crossing edge retains its edge ID and outside endpoint while
the owned endpoint is rebound to the declared root port. Missing, duplicate,
stale, ambiguous, unpinned, or value-changing receipts remain visible blocked
candidates. Preview responses expose hashes and dispositions, never the
supplement's prompts or parameter values.

A collapsed legacy Cluster may persist root port or preview metadata whose
execution child is intentionally absent from the saved snapshot. In that one
case, both client and backend reproduce the historical child ID from the root
ID and an exact, unique pinned V2 `semanticRole`; they never infer a target
from a label, node order, pipeline class, or model family. A missing or
duplicate semantic role remains blocked.

`POST /studio/composite-migrations/apply` is the only apply route. A caller
must copy the exact preview identity and literal confirmation:

```json
{
  "migrationId": "block-v2-migration-0123456789abcdef01234567",
  "planHash": "sha256:...",
  "confirmation": "APPLY_BLOCK_V2_MIGRATION",
  "allowBlockedCandidates": true,
  "compilerSupplement": {}
}
```

The server recomputes the plan under the shared workflow/User Node persistence
lock. Stale source bytes return `409`; missing authority returns `403`.
`allowBlockedCandidates` is required when a reviewed plan contains blocked
Cluster or malformed candidates and means “apply only the explicitly listed
safe targets,” not “convert the blocked records.” No route silently deletes,
renames, or merges a record. When the preview used a compiler supplement, apply
must resend the exact supplement; omitting or changing it recomputes a different
plan and returns `409` before any journal or workflow write.

The Setup → Advanced diagnostics migration card has two supported supplement
paths. **Generate exact supplement** first refreshes the backend preview,
reads the referenced saved workflow documents, and runs the same hidden
registered Block V2 compiler/finalizer used by catalog insertion. It preserves
the generated JSON for inspection, POSTs that exact parsed object for a second
read-only preview, and resends the identical object only after explicit Apply
confirmation. Progress and cancellation are visible; a workflow switch aborts
the generation, compiler transients are always removed, and dirty workflow
tabs warn that generation uses saved backend bytes rather than unsaved canvas
state. The JSON textarea remains available for an externally produced exact
supplement. Neither path auto-applies.

Visible generation accepts either an exact-current receipt or one exact
checked-in historical semantic-equivalence authority. The authority ledger is
`data/legacy-cluster-semantic-equivalence-receipts.v1.json`; it is data-only,
strictly hashed, and currently intentionally empty. No receipt was inferred or
manufactured for the saved historical instances.

Each future receipt has this exact semantic shape (hash values abbreviated):

```json
{
  "id": "legacy-cluster-equivalence:...",
  "historical": {
    "manifestDefinitionId": "...",
    "libraryRevision": "40-character commit",
    "manifestContentHash": "sha256:...",
    "executionAdmissionId": "...",
    "studioExecutionSpec": {
      "id": "...",
      "contentHash": "...",
      "executionProfileId": "..."
    },
    "executionGraphHash": "sha256:...",
    "interfaceHash": "sha256:..."
  },
  "destination": {
    "manifestDefinitionId": "...",
    "libraryRevision": "40-character commit",
    "manifestContentHash": "sha256:...",
    "executionAdmissionId": "...",
    "blockDefinitionId": "...",
    "blockDefinitionContentHash": "block-definition-v2-...",
    "blockDefinitionCanonicalSha256": "sha256:...",
    "executionGraphHash": "block-graph-v2-...",
    "interfaceHash": "block-interface-v2-..."
  },
  "review": {
    "decision": "semantic_equivalent",
    "issuer": "...",
    "reviewedAt": "offset ISO-8601 timestamp",
    "notes": "..."
  },
  "receiptHash": "sha256:..."
}
```

Receipt IDs, receipt hashes, and historical tuples are unique. Multiple
independently reviewed historical identities may bind to the same exact
destination. Both receipt and ledger hashes cover the complete review body. The
client receives only a value-free authority summary, selects the exact pinned
destination route, compiles in an isolated transient using destination
authority, and adds only `{receiptId, receiptHash}` to the per-instance
supplement. The backend reloads the checked-in receipt and verifies its exact
historical manifest/admission/Studio-spec tuple plus the destination manifest,
`BlockDefinitionV2` content/canonical hashes, graph hash, and interface hash.
It then applies the existing exhaustive per-instance node/value/port/preview/
topology mapping checks. A receipt is review authority, not mutation authority.
The UI renders the old-to-new hash bindings, issuer, timestamp, and notes as a
read-only diff; Apply still requires the literal confirmation and the exact
supplement, and rollback still restores the byte-exact journal backup.

A historical manifest without a reviewed archived definition/equivalence
receipt and a Cluster with no execution receipt remain unchanged with explicit
diagnostics. The client never aliases an old manifest to a current route by
name. The final backend preview remains the source-byte,
canonical-definition-pin, mapping, and topology authority. The read-only
inventory CLI does not produce conversion authority and MUST NOT be used as a
substitute.

The read-only 2026-09-02 local inventory currently contains 375 legacy
registered Cluster instances across 102 immutable manifest identities.
Twenty-one instances match two current manifest/admission identities and
their Studio execution-spec tuples exactly. Of the remaining 354 instances,
343 retain complete admission plus Studio execution-spec identities and 11 do
not. A subsequent byte-exact retained-artifact review recovered seven complete
historical definition bodies: six bodies have exact saved admission/Studio-
spec agreement for 42 instances, while the HunyuanVideo 1.5 body covers three
instances but has no historical execution admission. The read-only audit now
classifies the six execution-complete identities as archived-manifest exact;
312 instances across 94 identities still require historical evidence. The two
newly historical Qwen instances require a separate reviewed mapping or
equivalence receipt after the current Qwen route reseal.

The schema-v4 recovery audit records the evidence axes independently instead
of collapsing them into one generic blocker. For every immutable identity it
reports the current-catalog comparison, embedded-manifest completeness,
execution-receipt completeness, presentation-projection counts, registered
archive/equivalence status, recovered Studio-spec partial evidence, precise
missing-evidence codes, and safe next actions. It still returns no workflow
path, instance ID, prompt, or parameter value. In the current local source
review:

- all 375 roots embed only `{id, libraryRevision, contentHash}` and zero embed a
  complete canonical definition body;
- 364 roots retain a complete admission/Studio-spec tuple, while 11 roots
  across three identities retain neither;
- nine expanded roots retain 63 presentation children, but those children do
  not authenticate the original boundary, conditionals, or execution graph;
- `data/studio/composite-migrations/` is absent, so there are zero migration
  journals and zero byte backups to inspect;
- the checked-in semantic-equivalence ledger contains zero receipts;
- the checked-in archived-definition ledger retains seven canonical bodies,
  registers six bodies for the read-only audit, and retains HunyuanVideo 1.5 as
  body-only with the explicit reason `historical_execution_admission_absent`;
  and
- none of the original 99 historical content hashes occurs in either Git history. The
  exact 3,334,820-byte 2026-08-24 live node-library response
  (`sha256:cddf3bd841f7243a3757883c1fba42d78b404cf6bc78b7b4893365db5d07ba91`)
  contains none of them. Seven complete bodies instead survive in later
  retained local live-node-library transcript records dated 2026-08-25 through
  2026-08-29. The checked-in ledger stores only their relative transcript
  record locations, byte counts, SHA-256 receipts, and public definition
  bodies—not raw transcript contents. Its complete canonical receipt is
  `sha256:dd57906b8ae746ff1d74cda1b8e1c145c1b589cbbf0d218ab3011e705db47a90`.

Two exact, ignored frontend qualification captures also retain 18 distinct
historical Studio execution-spec bodies. Their checked-in, SHA-256-wrapped
partial-evidence ledger matches 20 immutable manifest identities, 21 exact
manifest/admission/Studio-spec tuples, and 65 saved instances. The extractor
reads only `/nodes/studioModelCapabilities/*/studioExecutionSpecs/*`; it does
not copy full browser state, workflow names/paths, instance IDs, prompts,
parameter values, or media into Git. Each recovered body's historical
`studio-spec-v1-*` self-hash is rechecked, but that 32-bit FNV receipt is weak
integrity evidence—not manifest, artifact, compiler, execution, or semantic-
equivalence authority. The separate partial-review ledger is initially empty
and permits only `verified_partial_evidence` or `rejected_evidence`; either
decision remains explicitly non-authorizing.

The original 99 historical identities have a current catalog entry with the
same definition ID and upstream library revision but a different manifest
content hash. The resealed current Qwen route adds one more non-current saved
identity covering two instances. That overlap is useful review context, never
compatibility authority.
The six registered archived bodies improve the read-only evidence disposition;
they do not themselves provide mutation authority. Three checked-in compiler
mappings cover 15 of the 354 non-current instances across three immutable
identities. Those instances are mapping-ready for exact supplement generation;
the remaining 339 instances across 97 identities are still blocked. All 354
workflow instances remain byte-unchanged until the existing preview/compiler-
supplement/apply path receives exact per-instance preservation evidence and an
explicit Apply confirmation. This limitation does not block the 49
valid V1 reusable User Node definitions or their 46 workflow instances, whose
complete embedded graphs are handled by the additive V1-to-V2 migration.

Historical compiler mappings use the independent, strictly validated
`data/legacy-cluster-compiler-mappings.v1.json` ledger. A record binds one
registered archived body and its exact manifest/admission/Studio tuple to one
exact current manifest/admission and BlockDefinitionV2 content, canonical,
graph, and interface identity. Duplicate identities, archive drift, unknown
records, destination drift, and altered mapping hashes fail closed. The client
compiler supplement carries only the reviewed mapping ID/hash; preview and
Apply resolve the checked-in record again and require the generated V2 graph
and interface to match it exactly. Mapping authority is mutually exclusive
with semantic-equivalence authority. It does not bypass explicit preview,
literal Apply confirmation, byte backup, refresh persistence, recovery
journaling, or rollback. The ledger currently contains three exact mappings:
Wan 2.2 TI2V 5B, MiniMax Music 3, and Transformers CTC speech recognition.
They cover 15 saved historical instances, which are now eligible for an exact
compiler-supplement preview. Qwen and two SDXL mappings remain blocked until
historical instance values are proven not to alter the independently compiled
current destination schema; Hunyuan remains body-only.

For archived compiler mappings, every preservation mapping must target
`BlockInstanceV2.values`. A historical value is not allowed to target a graph
parameter, alter `definitionSnapshot`, or produce an `effectiveGraph` different
from the frozen registered definition graph. This keeps prompts and compatible
parameters instance-local and prevents one historical Cluster from publishing
dynamic schema for another instance. Values without an exact declared current
input/control target fail closed. Qwen and the two SDXL routes require this
clean-current-definition plus separate-value-adapter flow before their pending
mappings can be reviewed.

`scan_registered_cluster_recovery_audit()` exposes the same distinction as a
deterministic schema-v4 aggregate report without workflow paths, instance IDs,
prompts, or parameters. Its summary adds mapping-eligible instance/identity
counts and remaining-blocked historical instance/identity counts. These are a
fail-closed, disjoint accounting of all non-current historical instances:
currently `354 = 15 mapping-ready + 339 still blocked`. Mapping presence alone
never authorizes conversion or workflow mutation. A complete archived node-library definition may be
supplied to that audit. By default it loads only the six execution-complete
bodies from `data/legacy-cluster-archived-definitions.v1.json`; the seventh
body-only record remains non-registered and non-convertible. An embedded body
is reported as unreviewed and does not change recoverability until explicitly
registered. A historical content hash, current definition with the same name,
or expanded presentation projection never changes recoverability.
`GET /studio/composite-migrations/recovery-audit` returns this read-only report.
Its `partialStudioSpecEvidence` section exposes only validated matched public
spec bodies and source byte receipts, while every evidence/review record and
the top-level boundary state `authorizesConversion: false`. It is not accepted
by migration preview, compiler supplementation, Apply, rollback, or semantic-
equivalence validation.

Each `identities[]` entry contains exact `executionTuples[]`. A tuple's
`recoveredStudioSpec.status` is either `missing` or
`self_hash_valid_partial_evidence`; its `manualReview.status` is
`evidence_missing`, `unreviewed`, `verified_partial_evidence`, or
`rejected_evidence`. Reviewed rows include the checked-in review ID, reviewer,
timestamp, notes, and hash. These fields never change the identity's existing
`disposition`. The summary separately reports recovered body, manifest-
identity, exact execution-tuple, and instance counts plus verified/rejected
partial-review tuple counts. `localEvidence` reports the complete checked-in
ledger counts and hashes even when a body has no matching saved instance;
`partialStudioSpecEvidence` contains only bodies and reviews matched by the
current inventory.
The separate `huggingface-cluster-catalog-gates.v1.json` ledger binds all 28
currently catalog-only definitions to seven exact artifact/legal/runtime
reviews. Its eligible route count is zero; it is evidence for keeping routes
closed, not a substitute for an execution admission.

Run both read-only audits without writing a report file:

```bash
.venv/bin/python scripts/audit_huggingface_cluster_backlog.py --data-dir data
```

Reproduce the bounded Studio-spec evidence JSON from retained local captures
without writing repository files:

```bash
.venv/bin/python scripts/extract_legacy_cluster_studio_spec_evidence.py \
  --data-dir data \
  --source frontend-capture-2026-08-24-sdxl=/path/to/inserted-state.json.gz \
  --source frontend-capture-2026-08-25-smollm2=/path/to/configured-state.json.gz \
  --compact
```

Both gzip byte sizes and compressed/uncompressed SHA-256 receipts are pinned;
unknown, changed, oversized, malformed, or symlinked captures fail closed. The
command emits canonical JSON to stdout only. Raw captures remain ignored and
must not be committed.

A historical compatibility receipt must pin the complete source tuple
(`definitionId`, library revision, manifest hash, admission, Studio execution
spec/hash, and execution profile) and its destination `BlockDefinitionV2`. It
must be accompanied by an exact compiler supplement that records semantic
node, value, port, preview, and topology mappings and passes representative
persistence and execution tests.
Presentation children, matching labels, matching model IDs, and a current
catalog definition are never migration authority. A self-contained
materialized legacy graph may instead be preserved as a non-registered User
Node after a strict completeness check; a missing-receipt or incomplete
instance stays fail-closed.

Before the first replacement, the backend saves the exact original bytes for
every target below
`data/studio/composite-migrations/{migration_id}/backups/` and records all
before/after hashes in `manifest.json`. Same-directory temporary files and
atomic replacements are used per target. An identical completed apply is
idempotent. For supplemented conversions, the manifest also records the
value-free supplement hash and exact compiler identity/node/edge absorption
receipts; idempotent apply still requires that same supplement. An interrupted
multi-file apply is reported for recovery rather than resumed implicitly.

`GET /studio/composite-migrations` lists transaction status and
`GET /studio/composite-migrations/{migration_id}` reports each target as
`source`, `applied`, `missing`, or `conflict`, plus an aggregate `applied`,
`rolled_back`, `interrupted`, or `conflict` state. Rollback requires:

```json
{
  "confirmation": "ROLLBACK_BLOCK_V2_MIGRATION"
}
```

sent to
`POST /studio/composite-migrations/{migration_id}/rollback`. Rollback verifies
every backup and current target hash, is idempotent, and refuses to overwrite
an independently changed or missing file; there is no force option. A rolled
back plan can be applied again with the same explicit authority after its
sources are revalidated.

Setup → Advanced diagnostics now contains the recovery/preview UI. It strictly
validates every preview, candidate, exact target path, hash, journal, target
state, and mutation receipt before display. Preview/list requests are GET-only
and never apply automatically. Apply opens an explicit review dialog showing
the exact target and blocked paths; it remains disabled until the operator
types `APPLY_BLOCK_V2_MIGRATION` and, when blocked candidates exist, separately
opts into converting only the listed safe targets. Rollback first refreshes the
exact journal and requires `ROLLBACK_BLOCK_V2_MIGRATION`; there is no force
control. HTTP `403`, stale-plan/state `409`, and backend validation failures
remain visible in the card and notification, and preview plus journal state is
refetched after every successful action. Mocked client tests exercise the
parsers, request bodies, error messages, review gates, exact-path rendering,
and recovery list without invoking a real migration. A disposable browser test
covers generate → final preview → Apply → browser refresh → list/status →
rollback against a dedicated loopback backend and a temporary exact copy of one
mapping-ready single-root workflow. It verifies the original SHA-256 before
Apply and byte-for-byte equality after rollback; the source data directory is
never served or mutated. Run it from `MoDiff-client` with
`npm run e2e:migration:isolated`. Backend
registered-Cluster compiler handoff, validation, application, backup,
rollback, and HTTP coverage are complete. Transaction logic is in
`modiff/composite_migration.py`; focused conversion, transaction, conflict,
rollback, HTTP, and storage-lock tests are in
`tests/test_composite_migration.py` and
`tests/test_composite_migration_server.py`; the client contract/UI tests are in
`MoDiff-client/scripts/composite-migration-ui.test.mjs`,
`MoDiff-client/scripts/registered-block-v2-insertion.test.mjs`, and
`MoDiff-client/tests/e2e/studio-mocked/studio-mocked.spec.ts`.

`GET /model_capabilities` publishes reviewed but unqualified generic adapters
under `experimentalCapabilities`. A record whose `qualificationStatus` is
`contract_only` identifies one exact registered Diffusers pipeline class and
execution kind, its generic `backendPath`, exact `runnableModes`, reviewed
upstream workflow contracts, backend-owned parameter aliases, and mode input
contracts. Immutable `defaultRepo`/`revisionCandidates` appear only after a
separate artifact admission; pure upstream contract discovery deliberately
omits them.
It also reports `autoEligible: false`, `templateEligible: false`, and
`galleryEligible: false`.

Contract-only means that the loader/action and reviewed state-flow or fake-call
contract, as applicable, are present; it is not evidence of a completed model
run. These records have no
`executionProfiles` or `optionalRuntimeRequirement`, do not enter the primary
supported `capabilities` list, and cannot be selected by Auto. Expert users can
still inspect the same generic node fields from `/nodes`. Templates, resource
qualification, live media, and Gallery publication require later graph and
remote qualification gates.

### Studio execution specifications

For migrated exact pairs, `GET /model_capabilities` publishes a
`studioExecutionSpecSchemaVersion: 1` marker, a bounded
`studioExecutionSpecModes` list, and one or more `studioExecutionSpecs` beside
the pair's `executionProfiles`. The mode list must match the specification modes
exactly: claimed modes fail closed when their specification is absent or
malformed, while unclaimed sibling modes remain on the reviewed migration path.
The root response also includes the same specification catalog. Each
specification binds one exact model/mode pair to its execution-profile ID,
loader module/action,
execution path, pipeline class, default repository, generic graph roles and
positions, typed edges, form bindings, ordered dynamic actions, and declared
Auto override fields. `contentHash` is the canonical
`studio-spec-v1-<8 lowercase hex>` checksum of every semantic field.

The server validates every node, parameter, input/output handle, connection
type, binding, and graph component against the live `/nodes` registry before
publishing the response. Unknown, incompatible, duplicate, or disconnected
contracts fail the capability request instead of falling back to a client
recipe. A managed submission includes a bounded
`runtimeHints.studioExecutionSpec` receipt containing exactly
`schemaVersion`, `id`, `contentHash`, and the specification role-to-node-ID
map. Graph admission checks that those exact nodes are executable and that all
declared edges and bindings remain present. The receipt and checksum are
consistency identifiers, not authorization tokens or live-model evidence.

An execution profile may also publish `expert_cuda_policy` with
`schema_version: 1`. This bounded policy declares the CUDA dtypes that Studio
must block, the recommended replacement dtype, the projected offloaded and
resident VRAM budgets, and exact per-quantization resident overrides. The
policy is optional: when it is absent, the client does not infer one from a
model family or pipeline name. Studio consumes it only from the unique
execution profile named by the selected exact specification. Auto admission
continues to use the backend resource plan rather than this Expert-facing
estimate.

An execution profile may additionally publish
`expert_quantization_policy` with `schema_version: 1`. The policy declares the
exact Expert quantization and offload modes plus the generic Modular Diffusers
quantization node and its reviewed component configuration. Studio applies the
policy only when the selected specification names that exact execution profile;
direct execution paths continue to derive their required loader fields from the
specification bindings. Missing, malformed, or registry-incompatible policy
data never falls back to a model-family or pipeline-name rule.

An execution profile may publish `expert_quantization_modes` as a bounded,
unique list drawn from `bnb_4bit`, `bnb_8bit`, `quanto_float8`, and
`torchao_float8`. Studio exposes only those choices for the exact selected
model-and-mode specification; absence means that no Expert quantization
selector is advertised. The client does not infer choices from a model family
or pipeline name. This field proves schema admission, not package delivery or
live qualification. See the [quantization support
matrix](quantization-support.md) for the current app-installable boundary.

An execution profile may also publish `expert_mps_policy` with
`schema_version: 1`, a reviewed `qualification` (`unqualified` or
`experimental`), and a bounded fallback action. Studio presents this advisory
only in Expert mode on Apple MPS and only when the selected exact specification
names that profile. The policy remains non-blocking and is omitted for profiles
without a reviewed MPS advisory.

The current schema-v1 catalog covers the migrated Flux Schnell, Dev, Krea,
Flux2 Klein, Depth, Canny, Redux, Kontext, and Fill image pairs; Wan 2.2 I2V
and TI2V; Wan 2.1 text, video, and color-edit modes; all four LTX condition
modes; all four advertised Wan VACE modes; all four ACE-Step audio modes;
direct Qwen Image and Z-Image text-to-image; Qwen Image Edit direct inpaint,
direct outpaint, and Modular `edit_image`; and Qwen Image Edit Plus Modular
`edit_image` and `multi_image_reference_edit`; Qwen Layered Modular
`layer_decomposition`; and Qwen Image Modular `control_image`. This covers all
39 currently declared execution-profile pairs. Related pairs reuse generic
image, video, or audio topologies while keeping distinct exact profiles,
artifacts, resource policies, form bindings, and receipt identities.

The generic Diffusers image loader's schema-v1 signal also contains the exact
reviewed field overlay for its selected pipeline class and mode. Connected
generic Generate, Edit, Inpaint, and Control Generate nodes validate that whole
signal before updating optional control visibility. Switching the loader value
therefore refreshes the same generic node; clients do not infer image controls
from a model or pipeline name, and a stale or edited overlay fails closed.

ACE-Step text-to-audio,
variation, continuation, and repaint use the generic Diffusers runtime recipe, audio
loader/generator, source-audio loader where required, and audio export nodes;
continuation additionally seals loudness matching and joining. Repaint reuses
the source-audio route with its exact task and range bindings.
Qwen Image Edit inpaint seals the reviewed direct pipeline, source-image and
mask loaders, generic inpaint node, preview route, and exact form bindings.
Outpaint reuses that reviewed loader and generator while sealing the distinct
generated canvas/mask node and all boundary-placement bindings. Modular Edit
seals the reviewed Models Loader, prompt encoder, source-image VAE encoder,
denoiser, latent decoder, preview, typed route-state edges, and exact dynamic
form bindings issued for `QwenImageEditModularPipeline`.
Qwen Image Edit Plus reuses that reviewed seven-role Modular edit contract for
both advertised modes while publishing a distinct receipt per mode and binding
the `QwenImageEditPlusModularPipeline` profile and immutable default artifact.
Qwen Layered seals its seven-role source-image, prompt, VAE encode, denoise,
decode, and preview route with the reviewed layer-count and resolution bindings.
Qwen Image Control seals an eight-role graph with the pinned Qwen ControlNet
Union component loader, control-image adapter, typed ControlNet bundle, and
route-state chain through denoise and decode. Its Hub selector and immutable
revision are bound independently from the base Qwen Image artifact.
Wan VACE text-to-video seals the reviewed direct VACE profile and shared
quantization, runtime-recipe, video loader/generator, and export route. Video
inpaint and outpaint additionally seal the source-video normalization and
aligned-mask route with their distinct reviewed mask-growth policies.
Control-to-video seals its separate control-video loader, normalization route,
and exact width, height, and frame-count bindings without adding the source or
mask branches.

### Auto resource compatibility

Both planning endpoints gather runtime/model snapshots, evaluate candidates,
and serialize their responses off the HTTP event loop. A Model Manager batch
must not block health, library, or cancellation requests while planning is
pending. Plans do not authorize execution; submission still revalidates the
current runtime, artifacts, and graph admission.

For expanded Modular V2 execution, an explicit loader `workflow_id` is checked
against the immutable upstream workflow snapshot used by the catalog. The
loader passes that selection to the installed Diffusers workflow constructor
and exposes the complete selected component bundle to downstream Modular
steps. Synthetic fixed-sequence `default` identities are not forwarded as
upstream workflow names. Unknown families/routes are rejected; this structural
selection does not grant model access, license consent, Auto authority, or
publication qualification.

`POST /auto_resource/plan` and every item returned by
`POST /auto_resource/plans` use `schemaVersion: 2`. The top-level
`compatibility` object is the authoritative UI assessment for the current
model, operation, runtime, installed artifacts, accelerator, accessible
memory, and supported placement recipe. It contains:

- `state`: `ready`, `needs_model`, `needs_setup`, `expert_only`, or
  `unsuitable`.
- `severity`, stable `code`, concise `summary`, and explanatory `detail`.
- An optional structured `action` such as model install/repair, environment
  repair, Setup, or Expert review.
- `source: backend_auto_planner`.

Clients must not override this assessment with separate GPU-vendor, OS,
dedicated-VRAM, or shared-memory thresholds. A client waiting for this response
should report a pending compatibility check. Existing plan fields remain for
execution and backward compatibility.

`POST /huggingface/cluster/auto-authority` is the instance-scoped authority
step for a registered V2 Block after Auto has selected a ready exact recipe.
The exact request shape is `{schemaVersion: 1, instance, form}`. The backend
strictly validates the complete normalized `BlockInstanceV2`, resolves its
catalog definition, execution admission, immutable artifact, and planner result
again, and recomputes the public definition content hash, effective graph hash,
effective-interface hash, and interface-bound execution-parameter hash.
Client-supplied opaque hash text is not accepted as proof. Each routed compiler
output is pinned by both the public V2 FNV identity and a backend-owned SHA-256
of its canonical validated definition material.

The form must include every resource-impacting key (`width`, `height`, `steps`,
`numFrames`, `device`, `dtype`, `quantizationMode`, `autoOffload`, and
`offloadMode`), and any form field bound to a V2 control must equal that
instance value. A successful response is a short-lived, instance-local `auto`
receipt. Graph, public interface, value, artifact, recipe, runtime, admission,
canonical-definition, or resource-field substitution fails closed.
Structurally or publicly customized registered Blocks are intentionally not
eligible for Auto authority; they remain runnable in Expert/manual mode against
their concrete expanded graph. Auto qualification errors use an Auto-specific
message prefix so the UI does not misidentify them as Expert failures.
The current real frontend qualification is bounded to the exact Qwen Image
text-to-image route: its Expert and Auto runs completed against the pinned V2
definition and returned a frontend preview. This is evidence for that image
route, not a claim that current V2 video/audio routes or every registered
admission have completed a real model run.
Receipt `issuedAt` and `expiresAt` values use canonical UTC ISO-8601 spelling
with exactly millisecond precision (`YYYY-MM-DDTHH:mm:ss.sssZ`). The authority
producer must round-trip its completed receipt through the same strict
`BlockInstanceV2` validator used for saved workflows before returning it. A
successful HTTP response is not sufficient authority if the receipt cannot be
attached to, normalized with, and revalidated against the exact instance.

Every declared schema-version-2 Auto candidate, including
`selectedCandidate` and `nextCandidate`, carries one exact backend-owned loader
target:

- `autoResourceSchemaVersion` repeats the owning plan schema for durable
  history and retry binding;
- `executionProfileId` identifies the exact reviewed execution profile;
- `modelType` and `mode` identify the declared model/task pair;
- `loaderModule` and `loaderAction` identify the loader node contract;
- `executionPath` identifies the reviewed execution adapter; and
- `pipelineClass` identifies the profile's canonical pipeline implementation;
- `optionalRuntimeProfileIds` identifies the exact reviewed optional-runtime
  profiles owned by that execution profile; and
- `optionalRuntimeRequirement` binds their requirement schema, delivery mode,
  current-required flag, profile IDs, and execution-profile IDs; and
- `studioExecutionSpecContract`, when the pair is specification-owned, binds
  the versioned graph-specification ID, content hash, and execution-profile ID;
  and
- `modelDependencies` is an exact bounded list of `id`, `kind`, `repo`, and
  immutable `revision` receipts for auxiliary or internally loaded model
  artifacts required by that model/task pair. Pairs with none publish `[]`.

The response `modelRequirements` map is also exact-pair data. Its keys are
`<modelType>:<mode>`, and every value carries the same
`loaderModule`/`loaderAction`/`executionPath` target resolved from one unique
execution profile. A model-only aggregate is not returned because different
operations for one Studio model can intentionally use different loaders. Each
entry also carries the same exact `modelDependencies` receipt published on its
candidates.

For an Auto graph run, `runtimeHints.autoResourceCandidateId` must equal the
selected plan `id`, and `runtimeHints.autoResourceCandidates` must contain
exactly one same-ID entry with the same execution-affecting recipe and proof
state. Auto retry plans carry `candidateId`, `modelType`, `mode`,
`loaderModule`, `loaderAction`, `executionPath`, and `pipelineClass`; the worker
resolves `candidateId` back to that bounded candidate list and applies the
canonical candidate fields. It rejects missing, duplicate, stale, cross-pair,
cross-profile, unqualified, unsupported, or unreviewed retry candidates. When
no candidate-bound retry plan is supplied, Auto ignores a submitted
`resourceRetryModes` list and derives later offload modes from the selected
exact execution profile in canonical memory-pressure order. Expert mode may
still submit its bounded `resourceRetryModes` list. The worker also ignores
client `modelFamily` and `lowVramMode` classifiers; exact model/profile identity
and the selected recipe already carry the reviewed execution facts.

Immediately before Auto admission, the worker derives
`controlledArtifacts` from executable graph paths rather than trusting a
submitted receipt. Supported Modular, direct-image, and direct-audio LoRA nodes
are resolved through the exact Safetensors contract and contribute their
module/action, safe Hub-or-local content identity, adapter name, scale,
scheduler contract, replacement policy, and descriptor digest. Executable
Spandrel upscalers contribute their pinned Hub snapshot or rehashed local-file
identity, and the loader revalidates template-declared revision, size, and
SHA-256 immediately before loading. Soundtrack and lyric-video branches also
contribute the non-primary Diffusers pipeline's exact repository, immutable
revision, class, and descriptor digest; the selected primary loader is excluded
because its existing Auto artifact receipt already owns that identity.
Disconnected and no-op nodes do not contribute. The worker copies the derived
ordered list to the selected plan and every candidate before comparing them,
and a submitted `controlledArtifacts` field is discarded. Local absolute roots
are represented only by safe filenames/content digests and are not exposed in
public runtime events. A candidate whose readiness came from local history must
also have exact current schema-v8 history for this derived receipt; base-only
evidence is downgraded before Auto admission instead of being represented as
live proof. Qualification proof remains advisory for an otherwise valid
executable graph. Independently safe or passed candidates do not depend on that
local history check.

Plan application considers only executable loader IDs referenced by graph
`paths`. Direct loaders must already expose the profile's exact
`pipeline_class`; modular `ModelsLoader` nodes must already expose the exact
`model_type`. Module/action equality alone, disconnected nodes, and class-name
substring inference never authorize a rewrite. A plan that matches zero exact
executable loaders, or whose matching loaders expose none of its requested
resource fields, fails closed. An exact target whose requested fields already
have the planned values is a valid idempotent application.

Auto admission and retry failures use bounded, non-echoing messages with the
`auto_resource` category. Relevant stable codes include
`auto_resource_pair_undeclared`, `auto_resource_pair_mismatch`,
`auto_resource_candidate_mismatch`, and `auto_resource_target_mismatch`.
Clients should refresh Auto for these failures; structurally valid manual
configurations remain available through Expert mode. Controlled LoRA
resolution failures use the same bounded non-echoing envelope with stable code
`controlled_artifact_mismatch` and never copy a submitted repository or local
path into the public error.

Local Auto history version 7 binds successful and failed evidence to the Auto
schema version, exact execution-profile ID, loader/path/class identity,
optional-runtime profile and delivery contract, artifact revision, optimization
recipe, specification-owned graph contract, immutable model-dependency
receipt, ordered controlled-LoRA receipt, workload shape, and runtime hardware
fingerprint. Evidence from an
older history schema or a replaced execution, optional-runtime, graph, or
model-dependency/controlled-artifact specification is retained on disk for
inspection but cannot promote a current candidate to `live_proven`. Because
the planner does not infer graph-authored controlled blocks, a nonempty
controlled-artifact history receipt is deliberately not reused by a later
base-only plan.

A specification-owned Auto candidate requires the matching
`runtimeHints.studioExecutionSpec` receipt at execution. The receipt maps the
reviewed roles to the submitted node IDs; the backend then verifies the exact
profile, node identities, typed connections, and form bindings against the
current specification before any node executes. A legacy pair with no backend
graph specification remains outside this receipt claim.

### Optional model runtime metadata

Diffusers execution profiles identify their declarative dependencies in
`optional_runtime_profiles`. Auto plans, model capabilities, and file items from
`GET /listgraphs` publish the corresponding `optionalRuntimeProfileIds` and
`optionalRuntimeProfiles`; candidates carry the IDs, and the root
`GET /model_capabilities` response also publishes the complete profile catalog.
Each exact execution-profile record and each resolved Auto, capability, or
workflow item also carries `optionalRuntimeRequirement`, a version-1 object
with exactly these seven fields:

- `schemaVersion`: literal `1`;
- `delivery`: `base` or `optional_overlay`;
- `requiredNow`: whether this exact executable contract currently requires the
  app-owned overlay;
- `profileIds`: zero to 32 unique optional-runtime profile IDs;
- `executionProfileIds`: zero to 32 unique execution-profile IDs;
- `state`: `base_satisfied`, `missing`, `wrong_version`,
  `present_unqualified`, `staged`, `active`, `busy_recovery_only`,
  `restart_required`, `repair_required`, or `unavailable`; and
- `reason`: a bounded lowercase snake-case code.

An item with no registered execution contract may publish empty ID arrays only
with `delivery: base` and `requiredNow: false`. An `optional_overlay`
requirement has non-empty ID arrays and becomes runnable only in `active`.
Active means the current worker and catalog both report the overlay active and
every required profile reports `contractState: qualified` and
`cutoverReady: true`. Missing, malformed, ambiguous, duplicate, oversized, or
inconsistent execution/catalog metadata resolves to `unavailable`, not active.

The current composite Transformers + PEFT profile has a complete
source-controlled six-target wheel lock for its ten exact `stagedRequirements`.
Its effective response also reports `platform` and `machine`. Linux and Windows
x86-64 report `contractState: qualified`, `cutoverReady: true`, and available
install/activation actions. Linux ARM64, Windows ARM64, and both macOS targets
report `candidate_unqualified`, remain base-delivered, and expose no actions.
Each lock includes the exact
filename, official PyPI URL, SHA-256, byte size, Python target, platform, and
machine. Its metadata-only package status is `missing`,
`wrong_version`, or `present_unqualified`; unreadable distribution metadata
fails closed as `wrong_version` with `metadataState: unreadable`. These
observations do not install anything during Auto planning, discovery, browsing,
or workflow open. Every current Diffusers execution profile publishes its full
platform delivery table. Linux/Windows x86-64 resolve to
`delivery: optional_overlay` and `requiredNow: true`; pending targets resolve to
`delivery: base` and `requiredNow: false`. An unknown or duplicate target fails
closed rather than selecting an overlay implicitly.

`GET /runtime/optional-runtimes` returns the same profile catalog plus bounded
overlay state, staged-environment summaries, and a redacted active job summary.
`overlay.processLoadStatus` is one of `base`, `active`,
`busy_recovery_only`, `repair_required`, or `restart_required`. Status listing
does not perform a full overlay hash or import optional packages. A live
runtime mutation gate serializes all graph admission and field actions. After
that mutation completes, persistent recovery or restart status blocks only an
exact execution contract whose `optionalRuntimeRequirement.requiredNow` is
true; base-delivered graphs and field actions remain runnable.

`POST /graph` checks executable loader nodes referenced by `paths`, and
`POST /fields/action` checks its authorized loader module, action, and values.
The worker repeats the check immediately before execution and again before a
loader module import or field callback. `runtimeHints` are never authority for
this decision. A blocked HTTP execution returns `409` with the fixed keys
`error`, `category: optional_runtime`,
`error_code: optional_runtime_<state>`, `message`, `recovery_hint`, and
`optionalRuntimeRequirement`. A worker failure uses the same bounded object and
may add only task/node identifiers; it does not expose tracebacks, host paths,
process details, or loader diagnostics.

The optional-runtime mutation routes use exact JSON objects:

- `POST /runtime/optional-runtimes/install` requires
  `{ "profileId": string, "specDigest": "sha256:<64 lowercase hex>",
"consent": true }`.
- `POST /runtime/optional-runtimes/activate` additionally requires a bounded
  `environmentId`.
- `POST /runtime/optional-runtimes/rollback` requires
  `{ "consent": true }`.
- `POST /runtime/optional-runtimes/jobs/{job_id}/cancel` accepts an empty body
  or an empty JSON object. `GET` on the same job path is read-only.

Unknown or duplicate fields, non-object bodies, non-literal consent, malformed
identifiers/digests, oversized bodies, cross-kind jobs, and stale terminal jobs
fail closed. The current profile rejects install and activation with HTTP `409`
before reserving a lease, creating a job/staging directory, opening the
network, or starting a subprocess. A successful future activation or rollback
requires a worker restart; an unsupervised process remains
`restart_required`. It releases the completed mutation gate: base-delivered
work remains runnable, while `optional_overlay` work stays blocked until the
worker restarts into the qualified active environment.

### Saved workflows and media

The `/workflows/{workflow_id}` store is separate from the legacy file browser.
`PUT` validates the identifier and JSON payload, writes below the configured
data directory, and emits `workflow_updated`; `DELETE` emits
`workflow_deleted`. A `404` means the requested identifier has no saved record.

Workflow storage and response encoding run outside the HTTP event loop. The
shared Studio persistence lock still serializes writes with migration/rollback;
notifications remain coupled to completed writes even if the requesting browser
disconnects. `GET /workflows?view=summary` retains only bounded metadata in memory,
keyed by file identity/size/modification/change times. Changed, replaced, corrupt,
or deleted files are checked again; full graph snapshots are not cached here.

Model inventory discovery does not parse tokenizer vocabularies or tensor shard
index payloads to infer loader classes. Loader configuration metadata is still
read, and model installation/admission checks still validate the exact artifacts.
Batch Auto planning shares one lazily resolved optional-runtime catalog snapshot
within that request, not across requests or actual graph admission. None of these
optimizations installs packages, changes model defaults, or grants execution
authority.

The `/media/*` routes accept the same opaque managed file identifier returned
by `POST /file`. Probe returns normalized media metadata. Export and preview
may run bundled FFmpeg/image conversion and populate the ignored
`data/.media-exports` cache before returning a representation; callers should
treat them as potentially expensive even though the representation endpoints
use `GET`. They reject paths outside configured roots and return `422` for an
unsupported format or conversion failure.

`GET /media_assets` lists temporary media tracked by the runtime. `DELETE
/media_assets` accepts `scope` equal to `all_unpinned`, `task`, or
`older_than`; task cleanup also requires `taskId`, and age cleanup accepts
`olderThanHours`. Cleanup is refused while a graph is active and is not a
secure-erasure guarantee.

### Optional runtime optimizations

The optimization catalog is app-owned and compatibility-filtered; it is not a
generic package installer. Package profiles without complete immutable
artifact locks publish `canInstall: false` and `canEnable: false` and reject
install/activation before creating a job, lease, staged directory, network
request, or subprocess. Existing hashless environments are
`legacy_unqualified`, are never inserted into the worker import path, and may
only be deactivated to the base environment through the compatibility rollback
route.

Staged environments are promoted without replacing an existing destination and
are bound to the directory identity captured by the install lease. A bounded
durable promotion journal contains only the environment ID, phase, canonical
manifest/validation digests, and timestamp. Interrupted promotion is reconciled
under the same global install lease; ambiguous or malformed states fail closed
as repair-required and are not projected as runnable environments.

Runtime-only enablement changes local opt-in state when its capability permits
it; probing records only a compatibility result; qualification additionally
asserts that the exact workload output was reviewed. Public job and receipt
responses are fixed-schema projections and never return raw subprocess output,
commands, tokens, or absolute paths. These mutations remain trusted operator
actions. See [Optional runtime optimizations](optional-runtime-optimizations.md)
for the qualification boundary.

`GET /model_artifact_catalog` returns the checked immutable Hugging Face model
catalog. `?refresh=1` performs live Hub metadata lookup for the optional
`modelType` or `repo` filter; it does not turn an unreviewed repository into a
supported model.

### Node registry

`GET /nodes` returns:

```json
{
  "instance": "backend-instance-id",
  "nodes": {
    "modules.Image.Load": {
      "module": "modules.Image",
      "action": "Load",
      "label": "Load Image",
      "params": {}
    }
  }
}
```

The actual `params` schema is node-defined and can include display metadata, supported values, dynamic actions, input/output types, and model selectors. Use the live registry rather than hardcoding a parallel node schema.

### Graph execution and queue state

`POST /graph` accepts the API graph exported by the client. A submitted `sid` associates WebSocket events with the initiating session. A successful response includes a generated `task_id`; it means the graph was queued, not that execution succeeded.

Visual loop metadata may set `durable: true` only when its `Loop Result`
returns a retained file-backed video asset. Durable loops require both
`workflowTabId` and `runInputHash` in `runtimeHints`. After each successful
iteration the backend atomically checkpoints bounded asset metadata under the
runtime data directory; a replacement worker can resume the same exact input
without regenerating completed segments. Checkpoints reject in-memory media,
files outside MoDiff's retained-media directory, malformed identities, and
missing retained files. Successful graph completion removes its checkpoint;
failed or interrupted runs retain it for an exact-input retry.

Workflow-owned asynchronous requests should include `workflowTabId` and the
non-negative integer `workflowCanvasEpoch` in graph `runtimeHints` and in
`POST /fields/action`. Dynamic `node_definition`, `set_field_visibility`,
`set_field_value`, and `set_field_params` events echo these as
`workflow_tab_id` and `workflow_canvas_epoch`; graph events also carry their
task/client-run identity when available. Queued field-action completion uses
the same envelope. These messages are sent only to the originating WebSocket
session. Clients must ignore a field/schema mutation when its session,
workflow, run identity, or canvas epoch no longer owns the visible document.
The extra fields are additive so older single-document clients remain wire
compatible.

Client callers must also choose the correct local ownership scope. A normal
visible form edit is form-scoped and must reject a response after the form
epoch advances. A hidden registered-Block compiler action is canvas-scoped:
it may complete while its transient fields advance the form epoch, but it must
still reject a workflow/canvas replacement. Compiler callers propagate action
errors to the atomic insertion transaction; ordinary visible field callers
report the error in the UI without creating an unhandled promise. Hidden
compiler nodes and their incident edges are transient and never participate in
Auto policy, persistence, history, API export, or canonical definition hashes.

Diffusers audio loaders publish an exact schema-versioned `audio_contract`
signal for the selected pipeline class and task mode. Its `fieldParams` member
is the reviewed field overlay for the generic audio `Generate` node, including
visibility, required state, task choices, and duration bounds. The receiving
field action reconstructs the canonical contract and requires an exact match
before emitting `set_field_params`; it does not trust a stored or client-edited
overlay. Clients apply that backend-authored update generically and must not
derive audio fields from pipeline or model names.

`GET /queue` is the reconnect-safe task snapshot. It includes queued work, the
current task, structured node/phase progress when available, and a bounded set
of compact recent terminal receipts. Current and queued graph runs retain the
complete workflow snapshot needed for immediate restoration. Completed
workflow snapshots and run outputs are loaded on demand through
`GET /runs/{task_id}` instead of being repeated in every queue poll.
Completion, cancellation, and failure are distinct terminal states.

Use the WebSocket for live progress and `GET /queue` to restore state after reconnect. Do not infer success only from an HTTP `200` returned by `POST /graph`.

### Studio preview state

Studio output history and the current preview are separate persisted concepts. `GET /studio_outputs` returns `outputs`, a monotonic `revision`, and `previewSlots`. A slot is scoped by `workflowTabId`, `nodeId`, and `fieldKey`; it carries `currentOutputId`, pending run identity, generation, status, and update time. Clients must use `currentOutputId` as the current-preview authority and treat the other matching records as history. They must not infer the current output from list order, a browser tab transition, or a saved canvas value.

Successful `POST /graph` admission marks only generated preview fields present in that submitted graph as pending and returns `preview_slots` with `preview_state_revision`. The matching `task_queued` WebSocket event carries the same state for other connected clients. A generated `update_value` atomically persists its output and promotes it through `preview_slot`; a newer pending task cannot be displaced by a late output from the task ahead of it. Terminal events carry any failed, cancelled, or completed-without-output slot changes.

Deleting the current output clears its slot and never promotes an older history record. The backend retains every output referenced by a current slot even when applying normal history bounds. Version-1 output files are read by choosing the most recent scoped output as a one-time legacy current value; the next mutation writes the version-2 state.

### Errors

Most JSON failures include an `error` value and may include `message`, `category`, `error_code`, recovery guidance, task/node identity, runtime hints, memory state, or loader diagnostics. Error payloads are richer for graph execution than for older utility routes. Clients should preserve unknown fields and fall back to HTTP status plus human-readable text.

## File boundary

Relative file operations resolve against configured `[paths] work_dir`; runtime persistence normally lives under `[paths] data`. Browsing, previews, and streams resolve canonical paths and reject traversal, sibling-prefix tricks, and symlinks that escape their configured root. This containment is not user authentication; configure dedicated narrow directories and do not expose the service to untrusted clients.

Uploads are written under configured data subdirectories and share the configured HTTP request cap, which defaults to 1 GiB. A copied workflow-share preview is limited to 256 MiB. Public share responses expose their media URL, hash, filename, and content type without leaking backend absolute paths. Studio outputs, blocks, shares, planner history, and downloaded models are persistent local mutations even when initiated through the browser.

## Model and code trust

- `GET /local_models`, `GET /hf_cache`, and `GET /model_cache/diagnostics`
  accept `refresh=true`. Concurrent refresh requests join one model-index
  actualization, and filesystem-heavy filtering, artifact validation, and
  diagnostics scans run outside the HTTP event loop. Hub inventory and cache
  diagnostics results are coalesced and cached only within the exact completed
  index generation. A requested refresh never falls back to a result from an
  older generation: if validation of the new generation fails, the request
  fails instead of reporting a previously installed model as ready. App-owned
  Hub downloads, cache deletions, and incomplete-download cleanup advance the
  same generation after their mutation completes.
- Studio requests `GET /health`, `GET /runtime/optional-runtimes`, and
  `GET /model_capabilities` alongside model discovery. Hardware/package probes,
  optional-runtime inspection, capability construction, and JSON encoding for
  those control responses run outside the HTTP event loop; simultaneous
  identical requests share one build. The multi-megabyte default capability
  response is serialized and cached before the HTTP listener opens, so normal
  Studio discovery never performs its JSON encoding while liveness requests
  are in flight. The stable runtime identity is seeded from the startup
  hardware probe, while `GET /system_stats` remains the off-loop dynamic
  hardware-statistics endpoint.
- `POST /hf_token` validates a token and writes it in plaintext to ignored `config.ini`.
- `POST /hf_download` accepts a JSON object with `repo_id`, optional `sid`,
  `repair`, `repair_source_repo_id`, a `files` string list, and an optional exact
  lowercase 40-character commit `revision`. Concurrent requests for one
  repository may join only when both the immutable revision and file selection
  match. When `revision` is omitted for a repository in the reviewed artifact
  catalog, the server selects that repository's immutable catalog revision;
  uncataloged user-selected repositories retain their existing Hub behavior. It
  can consume substantial network, disk, RAM, and accelerator resources. Every
  new app-owned task first resolves the same plan returned by
  `GET /hf_download/plan?repo_id=...`, requires a known immutable byte count,
  reserves that remaining size across the active queue, and preserves at least
  64 GiB free on the cache volume. An unknown plan fails with HTTP 503 and a
  plan that does not fit fails with HTTP 507; neither path deletes older models
  or starts a snapshot download. App-owned model and Gallery snapshot payloads
  use the standard Hub HTTP path so request timeouts and retries remain bounded
  while completed cache blobs stay intact. The app runs at most two of those
  transfers in total. Ordinary downloads may share that window; a model repair
  waits for active payload transfers to drain and blocks new ones while it
  validates and may replace an invalid target partial. The process-global Hub
  transport setting is restored to its exact prior value after each shared or
  exclusive window drains.
- `GET /template_gallery/status` verifies an existing local Gallery against the
  app's immutable Dataset identity. `GET /template_gallery/plan` obtains and
  validates the pinned manifest, returns exact download/staging reservations,
  includes active model-download reservations, and preserves the same 64 GiB
  safety margin. `POST /template_gallery/install` accepts only `{}`, repeats
  that plan immediately before downloading, joins concurrent requests, shares
  the app's two-transfer limit, hashes every staged file, and atomically
  promotes the verified tree. It never deletes model-cache entries. Restart the
  backend after active downloads finish so a newly installed
  `/template-gallery/*` static tree is registered.
- `DELETE /hf_cache/{hash}` deletes selected cached model revisions.
- `POST /custom_modules/install` accepts a Git URL or local directory, places it under `custom/`, and refreshes the live registry. Imported custom code has the backend process's permissions.
- Modular Diffusers nodes may expose `trust_remote_code` for stored-graph compatibility, but repository Python and standalone component loading with remote code are rejected before upstream construction. Custom Modular pipeline and Dynamic Block execution is limited to an exact cached 40-character Hub commit whose canonical `modular_model_index.json` resolves to MoDiff-reviewed installed Diffusers pipeline/block exports and pinned official Diffusers or Transformers components. MoDiff revalidates the repository identity immediately before copying the reviewed metadata into a private content-addressed snapshot. Local mutable repositories remain preview-only, and neither a preview nor a persisted checksum grants repository-code authorization.

HTTP reads and mutations require a literal loopback destination and peer. Browser requests with an `Origin` header must also use a loopback `http` or `https` origin; CLI HTTP clients without an `Origin` header remain supported over loopback. WebSocket upgrades use the same destination and peer boundary, browser clients must send a loopback Origin, and native clients without one are accepted only over a loopback connection. The initial `welcome.recent` list uses the same compact receipts as `GET /queue`; full completed workflow snapshots remain available through `GET /runs/{task_id}`. The separate supervisor control server binds to `127.0.0.1` and likewise rejects non-loopback browser origins.

## Compatibility

Stable product routes such as `/graph`, `/queue`, `/studio_outputs`, and `/workflows/share` should remain compatible with the separate MoDiff-client repository. Python integrations should use the `modiff` package, and backend routes do not use a package-name prefix.
