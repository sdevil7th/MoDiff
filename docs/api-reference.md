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
| Node state       | `POST /fields/action`, `GET /cache/{node}/{field}[/{index}]`, `DELETE /cache`                                                                                                      | Run dynamic field actions and access/clear node cache values.                                                                               |
| Files and graphs | `GET /listdir`, `GET /listgraphs`, `GET /file`, `POST /file`, `GET /preview`, `GET /stream`                                                                                        | Browse the configured working directory, load/save graph files, upload media, and stream previews.                                          |
| Saved workflows  | `GET /workflows`, `GET/PUT/DELETE /workflows/{workflow_id}`                                                                                                                        | List, read, replace, or delete versioned workflow records below the configured data directory.                                               |
| Media I/O        | `GET /media/capabilities`, `/media/probe`, `/media/export`, `/media/preview`                                                                                                       | Inspect a managed media identifier or return a cached, converted download/browser preview through the built-in deterministic media tools.   |
| Runtime          | `GET /health`, `/runtime/status`, `/runtime/resources`, `/runtime/options`, `/system_stats`, `/runtime/gpu_processes`; `POST /runtime/gpu_cleanup`                                  | Read readiness, resource, option, and hardware state or request best-effort runtime cleanup.                                                 |
| Optimizations    | `GET /runtime/optimizations`, `/jobs/{job_id}`, `/receipts`; `POST /runtime/optimizations/install`, `/activate`, `/rollback`, `/enable`, `/probe`, `/qualify`                       | Stage, validate, select, roll back, and qualify optional app-managed runtime packages and record their local evidence.                       |
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
generic package installer. `POST /runtime/optimizations/install` stages one
known capability in an isolated optional environment and returns a job with
HTTP `202`. Activation or rollback can select an environment and request a
supervised worker restart. Enablement changes local opt-in state; probing
records only a compatibility result; qualification additionally asserts that
the exact workload output was reviewed. Installation, activation, rollback,
enablement, probing, and qualification all mutate local state and must be
treated as trusted operator actions. See [Optional runtime
optimizations](optional-runtime-optimizations.md) for the support boundary.

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
- Modular Diffusers nodes may expose `trust_remote_code`. Remote custom pipelines/blocks require explicit trust metadata and an exact 40-character commit revision; moving branches and tags are rejected.

HTTP reads and mutations require a literal loopback destination and peer. Browser requests with an `Origin` header must also use a loopback `http` or `https` origin; CLI HTTP clients without an `Origin` header remain supported over loopback. WebSocket upgrades use the same destination and peer boundary, browser clients must send a loopback Origin, and native clients without one are accepted only over a loopback connection. The initial `welcome.recent` list uses the same compact receipts as `GET /queue`; full completed workflow snapshots remain available through `GET /runs/{task_id}`. The separate supervisor control server binds to `127.0.0.1` and likewise rejects non-loopback browser origins.

## Compatibility

Stable product routes such as `/graph`, `/queue`, `/studio_outputs`, and `/workflows/share` should remain compatible with the separate MoDiff-client repository. Python integrations should use the `modiff` package, and backend routes do not use a package-name prefix.
