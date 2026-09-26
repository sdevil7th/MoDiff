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
