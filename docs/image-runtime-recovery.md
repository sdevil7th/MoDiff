# Image runtime pressure and recovery

The existing graph executor and process supervisor remain the only execution
owners. Recovery does not resume a partially completed denoising step.

- Reserve all managed components in a dispatch before loading any of them.
  Proactive accelerator eviction and CPU-pressure eviction must skip these
  leases, including nested dispatches.
- Discarding a model removes references; it must not materialize a full CPU copy
  of an offloaded model. CPU offload remains a separate, explicit operation.
- A cached node whose managed handle was evicted executes again, with the cache
  reason `models_evicted`. Successful unchanged runs still reuse resident models.
- Memory-observation failure stops that eviction pass. Cleanup errors are logged
  and must not replace the original execution failure.
- The memory manager never replays an opaque failed callback: its generator or
  pipeline may already be mutated. OOM reaches the graph's existing bounded retry
  policy, which discards failed-attempt caches and rebuilds declared inputs/seeds.
- Unexpected worker exits reconcile the durable queue: the active run fails and
  queued runs are cancelled with their workflow/navigation information retained.
  A fresh worker can accept new work; interrupted model code is not resumed.
- Five rapid worker failures stop automatic replacement. Delays are 1, 2, 4 and
  8 seconds and are interruptible by shutdown. A worker lasting at least 60 seconds
  resets this rapid-failure counter. Intentional forced-cancel replacements use
  the existing separate path. Fix the reported cause before manually restarting
  after exhaustion; the supervisor does not spin indefinitely on an import or
  device failure.

The campaign runner owns isolated backend processes and has separate per-job,
case, startup, request and teardown bounds. A terminal failed case is recorded;
later cases/routes continue where safe. An unknown or still-running model call
ends that route's owned process tree instead of overlapping another model run.
`--cases baseline,seed,guidance` selects focused verification and records
`selected_cases_only` in the receipt; it is not full modification coverage.

Fault-injection tests cover pressure, missing handles, nested leases, cleanup
failure, queue recovery and restart exhaustion without exhausting host RAM or
deliberately destabilizing the GPU driver. These are not Windows/ROCm/CUDA/MPS
hardware-OOM qualification. No software can guarantee recovery from every OS,
driver or hardware failure.

Auto also rejects a dense native placement when the exact installed safetensors
metadata proves that its parameter storage cannot fit the accelerator. Full
residency requires all selected components; model CPU offload still requires one
whole component at a time. Group offload uses a different residency contract.
These read-only dtype-aware lower bounds execute no model code and never prove
fit, activation demand, or hardware qualification. Unknown layouts retain the
existing requirement checks; a cached success cannot override an impossible
current placement.

Forced math attention on a native Modular loader requires an explicit working
memory budget when its Auto recipe otherwise declares only machine capacity.
Automatic attention remains available through the ordinary Diffusers API.
New Z-Image recipes select automatic attention; saved graphs retain their authored
policy and can use Custom memory. An earlier forced-math failure and a successful
automatic-attention run are different consumed recipes, not output parity proof.
Native flash SDPA requires compiled support in the installed Torch build. A CUDA
device alone does not establish that support. Capability discovery checks Torch's
build API, and explicit unsupported flash recipes fail before standard pipeline
loading. Fresh generic bindings use automatic attention unless the authored form
or reviewed candidate specifies a backend; saved recipes retain their policy.

For a reviewed image-conditioned Modular graph, inferred `edit_image` and
`multi_image_reference_edit` task aliases use their declared image-conditioned
resource mode, preferring the conservative multi-reference mode when offered.
This does not rewrite the graph or input order,
grant qualification, or admit explicitly conflicting task selections.
