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
| Models           | `GET /model_capabilities`, `/model_artifact_catalog`, `/model_fingerprints`, `/local_models`, `/hf_cache`, `/model_cache/diagnostics`, `/hf_hub`, `/hf_download/plan`; `POST /hf_download`, `/hf_token`; `DELETE /hf_cache/{hash}` | Discover, diagnose, space-plan, download, authenticate, fingerprint, and delete model artifacts. |
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
`contract_only` identifies one exact registered Diffusers pipeline class and
execution kind, its generic `backendPath`, immutable
`defaultRepo`/`revisionCandidates`, exact `runnableModes`, backend-owned
parameter aliases, and mode input contracts.
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
or pipeline name.

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
  or starts a snapshot download.
- `DELETE /hf_cache/{hash}` deletes selected cached model revisions.
- `POST /custom_modules/install` accepts a Git URL or local directory, places it under `custom/`, and refreshes the live registry. Imported custom code has the backend process's permissions.
- Modular Diffusers nodes may expose `trust_remote_code` for stored-graph compatibility, but repository Python and standalone component loading with remote code are rejected before upstream construction. Custom Modular pipeline and Dynamic Block execution is limited to an exact cached 40-character Hub commit whose canonical `modular_model_index.json` resolves to MoDiff-reviewed installed Diffusers pipeline/block exports and pinned official Diffusers or Transformers components. MoDiff revalidates the repository identity immediately before copying the reviewed metadata into a private content-addressed snapshot. Local mutable repositories remain preview-only, and neither a preview nor a persisted checksum grants repository-code authorization.

HTTP reads and mutations require a literal loopback destination and peer. Browser requests with an `Origin` header must also use a loopback `http` or `https` origin; CLI HTTP clients without an `Origin` header remain supported over loopback. WebSocket upgrades use the same destination and peer boundary, browser clients must send a loopback Origin, and native clients without one are accepted only over a loopback connection. The initial `welcome.recent` list uses the same compact receipts as `GET /queue`; full completed workflow snapshots remain available through `GET /runs/{task_id}`. The separate supervisor control server binds to `127.0.0.1` and likewise rejects non-loopback browser origins.

## Compatibility

Stable product routes such as `/graph`, `/queue`, `/studio_outputs`, and `/workflows/share` should remain compatible with the separate MoDiff-client repository. Python integrations should use the `modiff` package, and backend routes do not use a package-name prefix.
