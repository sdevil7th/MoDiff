# Local API Reference

MoDiff serves its bundled client and backend API from the same origin, normally `http://127.0.0.1:8088`.

This API is unversioned, unauthenticated, and intended for the trusted local MoDiff client. It is not a public multi-user API. Several routes mutate files, download models, import Python code, or execute graphs; read [SECURITY.md](../SECURITY.md) before writing another client or changing the bind address.

## Route groups

| Area             | Routes                                                                                                                                                                             | Purpose                                                                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Bundled client   | `GET /`, `/favicon.ico`, `/assets/*`, `/template-gallery/*`, `/user/*`, `/static/{module}/{file}`                                                                                  | Serve the generated frontend, proof gallery, and module UI assets.                                                                          |
| WebSocket        | `GET /ws`                                                                                                                                                                          | Session handshake, queue restoration, progress/events, field signals, and node updates.                                                     |
| Registry         | `GET /nodes`                                                                                                                                                                       | Return the live registered node contracts used by the bundled client.                                                                       |
| Execution        | `POST /graph`, `GET /queue`, `DELETE /queue/{task_id}`, `GET /stop`                                                                                                                | Queue, inspect, remove, or interrupt graph work. `GET /stop` is a legacy state-changing route.                                              |
| Node state       | `POST /fields/action`, `GET /cache/{node}/{field}[/{index}]`, `DELETE /cache`                                                                                                      | Run dynamic field actions and access/clear node cache values.                                                                               |
| Files and graphs | `GET /listdir`, `GET /listgraphs`, `GET /file`, `POST /file`, `GET /preview`, `GET /stream`                                                                                        | Browse the configured working directory, load/save graph files, upload media, and stream previews.                                          |
| Runtime          | `GET /health`, `GET /runtime/status`, `GET /system_stats`, `GET /runtime/gpu_processes`, `POST /runtime/gpu_cleanup`                                                               | Read readiness/hardware state and request best-effort runtime cleanup.                                                                      |
| Auto resource    | `POST /auto_resource/plan`, `POST /auto_resource/plans`, `GET /auto_resource/history`, `DELETE /auto_resource/history`                                                             | Plan hardware-aware model recipes and manage local planner history.                                                                         |
| Models           | `GET /model_capabilities`, `/model_fingerprints`, `/local_models`, `/hf_cache`, `/model_cache/diagnostics`, `/hf_hub`, `/hf_download`; `POST /hf_token`; `DELETE /hf_cache/{hash}` | Discover, diagnose, download, authenticate, fingerprint, and delete model artifacts. `GET /hf_download` performs a state-changing download. |
| Custom modules   | `GET /custom_modules`; `POST /custom_modules/refresh`, `/install`, `/{name}/update`, `/{name}/disable`, `/{name}/enable`                                                           | Clone/copy and import trusted custom Python modules or change their enabled state.                                                          |
| Studio outputs   | `GET/POST /studio_outputs`, `PATCH/DELETE /studio_outputs/{output_id}`                                                                                                             | Persist and manage local Studio output metadata and copied media.                                                                           |
| Studio blocks    | `GET/POST /studio/blocks`, `GET/DELETE /studio/blocks/{block_id}`                                                                                                                  | Persist reusable local graph blocks.                                                                                                        |
| Workflow shares  | `GET /workflow_shares`, `POST /workflows/share`, `GET /workflows/share/{share_id}`, `GET /workflows/share/{share_id}/media/{filename}`                                             | Create and render local workflow share packages and their copied preview media.                                                             |

## Core response contracts

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

`GET /queue` is the reconnect-safe task snapshot. It includes queued work, the current task, structured node/phase progress when available, and a bounded set of recent terminal receipts. Completion, cancellation, and failure are distinct terminal states.

Use the WebSocket for live progress and `GET /queue` to restore state after reconnect. Do not infer success only from an HTTP `200` returned by `POST /graph`.

### Errors

Most JSON failures include an `error` value and may include `message`, `category`, `error_code`, recovery guidance, task/node identity, runtime hints, memory state, or loader diagnostics. Error payloads are richer for graph execution than for older utility routes. Clients should preserve unknown fields and fall back to HTTP status plus human-readable text.

## File boundary

Relative file operations resolve against configured `[paths] work_dir`; runtime persistence normally lives under `[paths] data`. The server attempts to reject browsing and previews outside the working boundary, but this is not an authorization system. Configure a dedicated narrow directory and do not expose the service to untrusted clients.

Uploads are written under configured data subdirectories. Studio outputs, blocks, shares, planner history, and downloaded models are persistent local mutations even when initiated through the browser.

## Model and code trust

- `POST /hf_token` validates a token and writes it in plaintext to ignored `config.ini`.
- `GET /hf_download` can consume substantial network, disk, RAM, and accelerator resources.
- `DELETE /hf_cache/{hash}` deletes selected cached model revisions.
- `POST /custom_modules/install` accepts a Git URL or local directory, places it under `custom/`, and refreshes the live registry. Imported custom code has the backend process's permissions.
- Modular Diffusers nodes may expose `trust_remote_code`. Enable it only for reviewed, revision-pinned repositories.

## Compatibility

Stable product routes such as `/graph`, `/queue`, `/studio_outputs`, and `/workflows/share` should remain compatible with the separate MoDiff-client repository. Python integrations should use the `modiff` package, and backend routes do not use a package-name prefix.
