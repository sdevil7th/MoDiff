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

| Area             | Routes                                                                                                                                                                             | Purpose                                                                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Bundled client   | `GET /`, `/favicon.ico`, `/assets/*`, optional `/template-gallery/*`, `/user/*`, `/static/{module}/{file}`                                                                         | Serve the generated frontend and module UI assets. The Gallery route exists only for an explicit offline/local asset build; normal releases use an immutable public Hugging Face Dataset. |
| WebSocket        | `GET /ws`                                                                                                                                                                          | Session handshake, queue restoration, progress/events, field signals, and node updates.                                                     |
| Registry         | `GET /nodes`                                                                                                                                                                       | Return the live registered node contracts used by the bundled client.                                                                       |
| Execution        | `POST /graph`, `GET /queue`, `GET /runs/{task_id}`, `DELETE /queue/{task_id}`, `POST /stop`                                                                                        | Queue, inspect, remove, or interrupt graph work. A normally supervised backend replaces its worker when a blocking model call misses the cancellation grace period. |
| Node state       | `POST /fields/action`, `GET /cache/{node}/{field}[/{index}]`, `DELETE /cache`                                                                                                      | Run declared dynamic field actions and access/clear statically declared media or text cache fields. Connector, process-local, and other opaque outputs are not cache-servable. |
| Files and graphs | `GET /listdir`, `GET /listgraphs`, `GET /file`, `POST /file`, `GET /preview`, `GET /stream`                                                                                        | Browse the configured working directory, load/save graph files, upload media, and stream previews.                                          |
| Saved workflows  | `GET /workflows`, `GET/PUT/DELETE /workflows/{workflow_id}`                                                                                                                        | List, read, replace, or delete versioned workflow records below the configured data directory.                                               |
| Media I/O        | `GET /media/capabilities`, `/media/probe`, `/media/export`, `/media/preview`                                                                                                       | Inspect a managed media identifier or return a cached, converted download/browser preview through the built-in deterministic media tools.   |
| Runtime          | `GET /health`, `/runtime/status`, `/runtime/resources`, `/runtime/options`, `/system_stats`, `/runtime/gpu_processes`; `POST /runtime/gpu_cleanup`                                  | Read readiness, resource, option, and hardware state or request best-effort runtime cleanup.                                                 |
| Optimizations    | `GET /runtime/optimizations`, `/jobs/{job_id}`, `/receipts`; `POST /runtime/optimizations/install`, `/activate`, `/rollback`, `/enable`, `/probe`, `/qualify`, `/jobs/{job_id}/cancel` | Inspect runtime features and legacy package contracts, manage recovery, and record bounded local qualification evidence. Hashless package install and activation are unavailable. |
| Optional model runtimes | `GET /runtime/optional-runtimes`, `/jobs/{job_id}`; `POST /runtime/optional-runtimes/install`, `/activate`, `/rollback`, `/jobs/{job_id}/cancel` | Publish the reviewed optional-library contract and its fail-closed staged lifecycle. The current candidate exposes no executable install or activation action. |
| Auto resource    | `POST /auto_resource/plan`, `POST /auto_resource/plans`, `GET /auto_resource/history`, `DELETE /auto_resource/history`                                                             | Plan hardware-aware model recipes and manage local planner history.                                                                         |
| Models           | `GET /model_capabilities`, `/model_artifact_catalog`, `/model_fingerprints`, `/local_models`, `/hf_cache`, `/model_cache/diagnostics`, `/hf_hub`; `POST /hf_download`, `/hf_token`; `DELETE /hf_cache/{hash}` | Discover, diagnose, download, authenticate, fingerprint, and delete model artifacts. |
| Media lifecycle  | `GET /media_assets`, `DELETE /media_assets`                                                                                                                                        | Inspect temporary media records or remove exact unpinned, task-scoped, or age-scoped files while no generation is active.                   |
| Custom modules   | `GET /custom_modules`; `POST /custom_modules/refresh`, `/install`, `/{name}/update`, `/{name}/disable`, `/{name}/enable`                                                           | Clone/copy and import trusted custom Python modules or change their enabled state.                                                          |
| Studio outputs   | `GET/POST /studio_outputs`, `PATCH/DELETE /studio_outputs/{output_id}`                                                                                                             | Persist and manage local Studio output metadata and copied media.                                                                           |
| Studio blocks    | `GET/POST /studio/blocks`, `GET/DELETE /studio/blocks/{block_id}`                                                                                                                  | Persist reusable local graph blocks.                                                                                                        |
| Workflow shares  | `GET /workflow_shares`, `POST /workflows/share`, `GET /workflows/share/{share_id}`, `GET /workflows/share/{share_id}/media/{filename}`                                             | Create and render local workflow share packages and their copied preview media.                                                             |

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
- Python and required-package status.
- Registered module counts.
- Server and relevant configuration state.
- Queue summary.
- Normalized `hardware` data.

`GET /system_stats` returns the normalized hardware snapshot directly. The current schema includes `schema_version`, `system`, `torch`, `devices`, `default_device`, and `disk`. Callers should tolerate additive fields and individual probe errors.

`GET /runtime/resources` returns a short-lived schema-versioned sample of the
MoDiff process, storage roots, detected accelerators, active device, and current
run. Storage capacity remains available as `usedBytes`, `freeBytes`,
`totalBytes`, and `percent`; `activePercent` is interval I/O active time for the
disk backing the MoDiff data path. On Windows it is sampled from the physical
disk idle/query counters used by the operating-system performance telemetry.
Unsupported or inaccessible activity counters return `null` rather than
substituting capacity used. `GET /runtime/options` returns the live node option descriptors after
device and package compatibility filtering. Both are observations, not proof
that a real model workload completed.

### Contract-only Diffusers capabilities

`GET /model_capabilities` publishes reviewed but unqualified generic adapters
under `experimentalCapabilities`. A record whose `qualificationStatus` is
`contract_only` identifies one exact standard Diffusers pipeline class, its
generic `backendPath`, immutable `defaultRepo`/`revisionCandidates`, exact
`runnableModes`, backend-owned parameter aliases, and mode input contracts.
It also reports `autoEligible: false`, `templateEligible: false`, and
`galleryEligible: false`.

Contract-only means that the loader/action and fake-call contract are present;
it is not evidence of a completed model run. These records have no
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
The current schema-v1 catalog covers the migrated Flux Schnell, Dev, Krea,
Flux2 Klein, Depth, Canny, Redux, Kontext, and Fill image pairs; Wan 2.2 I2V
and TI2V; Wan 2.1 text, video, and color-edit modes; all four LTX condition
modes; and all four ACE-Step audio modes. Related pairs reuse generic image, video,
or audio topologies while keeping distinct exact profiles, artifacts, resource
policies, form bindings, and receipt identities. ACE-Step text-to-audio,
variation, continuation, and repaint use the generic Diffusers runtime recipe, audio
loader/generator, source-audio loader where required, and audio export nodes;
continuation additionally seals loudness matching and joining. Repaint reuses
the source-audio route with its exact task and range bindings.

### Auto resource compatibility

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

Every declared schema-version-2 Auto candidate, including
`selectedCandidate` and `nextCandidate`, carries one exact backend-owned loader
target:

- `executionProfileId` identifies the exact reviewed execution profile;
- `modelType` and `mode` identify the declared model/task pair;
- `loaderModule` and `loaderAction` identify the loader node contract;
- `executionPath` identifies the reviewed execution adapter; and
- `pipelineClass` identifies the profile's canonical pipeline implementation.

The response `modelRequirements` map is also exact-pair data. Its keys are
`<modelType>:<mode>`, and every value carries the same
`loaderModule`/`loaderAction`/`executionPath` target resolved from one unique
execution profile. A model-only aggregate is not returned because different
operations for one Studio model can intentionally use different loaders.

For an Auto graph run, `runtimeHints.autoResourceCandidateId` must equal the
selected plan `id`, and `runtimeHints.autoResourceCandidates` must contain
exactly one same-ID entry with the same execution-affecting recipe and proof
state. Auto retry plans carry `candidateId`, `modelType`, `mode`,
`loaderModule`, `loaderAction`, `executionPath`, and `pipelineClass`; the worker
resolves `candidateId` back to that bounded candidate list and applies the
canonical candidate fields. It rejects missing, duplicate, stale, cross-pair,
cross-profile, unqualified, unsupported, or unreviewed retry candidates.

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
configurations remain available through Expert mode.

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

The current composite Transformers + PEFT profile is contract metadata plus a
non-runnable staged-lifecycle scaffold. It reports
`contractState: candidate_unqualified`, `cutoverReady: false`, empty
`artifactLocks`, ten exact `stagedRequirements`, and unavailable install and
activation actions. Its metadata-only package status is `missing`,
`wrong_version`, or `present_unqualified`; unreadable distribution metadata
fails closed as `wrong_version` with `metadataState: unreadable`. These
observations do not change Auto selection, `canAutoRun`, or execution
readiness, and browsing or opening a workflow never imports, installs, or
activates the runtime. Every current Diffusers execution profile has
`delivery: base` and `requiredNow: false`; publishing an optional profile ID is
dependency metadata, not an activation gate. A future cutover must change the
authoritative exact execution profile to `optional_overlay` atomically.

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

- `POST /hf_token` validates a token and writes it in plaintext to ignored `config.ini`.
- `POST /hf_download` accepts a JSON object with `repo_id`, optional `sid`, `repair`, `repair_source_repo_id`, and a `files` string list. It can consume substantial network, disk, RAM, and accelerator resources.
- `DELETE /hf_cache/{hash}` deletes selected cached model revisions.
- `POST /custom_modules/install` accepts a Git URL or local directory, places it under `custom/`, and refreshes the live registry. Imported custom code has the backend process's permissions.
- Modular Diffusers nodes may expose `trust_remote_code` for stored-graph compatibility, but the current backend rejects all custom Modular pipeline and Dynamic Block execution, plus standalone component loading with remote code, before upstream construction. Exact cached 40-character commits may provide bounded declarative contract previews; a preview or persisted checksum is not execution authorization.

HTTP reads and mutations require a literal loopback destination and peer. Browser requests with an `Origin` header must also use a loopback `http` or `https` origin; CLI HTTP clients without an `Origin` header remain supported over loopback. WebSocket upgrades use the same destination and peer boundary, browser clients must send a loopback Origin, and native clients without one are accepted only over a loopback connection. The initial `welcome.recent` list uses the same compact receipts as `GET /queue`; full completed workflow snapshots remain available through `GET /runs/{task_id}`. The separate supervisor control server binds to `127.0.0.1` and likewise rejects non-loopback browser origins.

## Compatibility

Stable product routes such as `/graph`, `/queue`, `/studio_outputs`, and `/workflows/share` should remain compatible with the separate MoDiff-client repository. Python integrations should use the `modiff` package, and backend routes do not use a package-name prefix.
