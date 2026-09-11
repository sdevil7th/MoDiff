# Troubleshooting

Start with the backend preflight. It checks the runtime without importing every model node:

```bash
./scripts/with-runtime-env.sh ./.venv/bin/python -m modiff.preflight --json --check-port 8088 --fail-on-error
```

On Windows, use `.\.venv\Scripts\python.exe` in place of `./.venv/bin/python`.

For a human-readable summary, omit `--json`. Do not treat a successful preflight as proof that a particular model is installed, licensed, compatible with the host, or able to finish a generation.

If preflight reports `runtimeProfile.status: repair-required` with
`runtime-contract-drift`, the checked profile requirements, `pyproject.toml`,
or accelerator manifest changed after `.venv` was installed. Run the exact
`runtimeProfile.repair_command` shown in the report, then rerun preflight. Do
not edit `modiff-profile.json` or use a generic resolver to silence the check;
the installer recreates and validates the saved contract atomically.

## The backend does not start

1. Confirm Python 3.12 and the active executable reported by preflight.
2. Repair the managed environment, then verify its installed packages:

   ```bash
   ./install.sh --accelerator auto --backend-only --repair
   uv pip check --python .venv/bin/python
   ```

   On Windows, run `.\install.ps1 -Accelerator auto -BackendOnly -Repair`, then point `uv pip check --python` at `.venv/Scripts/python.exe`.

3. Look at the first error in the backend console. MoDiff logs to the console by default.
4. If an optional node package fails, follow that feature's documented managed-package guidance and restart so registry discovery runs again. Do not repair an accelerator profile with a generic `uv sync`.

### Port 8088 is already in use

Check the owner before stopping anything.

Windows PowerShell:

```powershell
Get-NetTCPConnection -State Listen -LocalPort 8088 |
  Select-Object LocalAddress, LocalPort, OwningProcess
Get-CimInstance Win32_Process -Filter "ProcessId = <PID>" |
  Select-Object ProcessId, Name, CommandLine
```

Linux/macOS:

```bash
lsof -nP -iTCP:8088 -sTCP:LISTEN
```

If the listener is an intended MoDiff backend, open it instead of launching a second process. Otherwise stop only the identified process, or change `[server] port` in `config.ini`. Do not kill every Python process by executable name.

## The UI is missing, stale, or incomplete

The backend serves generated frontend files from `web/`. Every build requires:

```text
web/index.html
web/assets/index.js
web/assets/index.css
web/favicon.ico
```

A normal remote-asset release intentionally has no
`web/template-gallery/`. An explicit offline/local Gallery build additionally
requires `web/template-gallery/manifest.json` and its referenced files.

With the backend running:

```bash
curl --fail http://127.0.0.1:8088/
curl --fail http://127.0.0.1:8088/assets/index.js
```

For an offline/local Gallery build, also run:

```bash
curl --fail http://127.0.0.1:8088/template-gallery/manifest.json
```

If one is missing or the UI does not match the separate client repository, rebuild MoDiff-client and mirror the **contents** of `dist/` into backend `web/` as described in [the root README](../README.md#updating-the-bundled-client). Copying the directory itself can accidentally create `web/dist/`, which the server does not use. Delete stale generated bundle files through the documented mirror, preserve backend-owned `web/user/` custom fields, restart the backend, and hard-refresh the browser.

For a remote build, verify that Gallery requests use the configured immutable
public Hugging Face Dataset revision. For a local build, an empty or
`unverified` Gallery manifest is not proof that template examples were
generated; keep the manifest and referenced media from one validated client
build.

The running app can also materialize or repair its exact byte-pinned payload
from **Setup → Template Gallery assets**. The action first validates the remote
manifest and refuses to start unless the full download plus staging copy,
active model-download reservations, and the 64 GiB safety reserve fit. It does
not delete cached models. After it completes, wait for active downloads to
finish, restart MoDiff, and then verify `/template-gallery/manifest.json`.

## CUDA is not detected

- Check `hardware.devices` and `hardware.torch` in preflight or `GET /system_stats`.
- Verify the NVIDIA driver can see the GPU with `nvidia-smi`.
- The managed NVIDIA profile targets reviewed PyTorch CUDA 12.8 wheels; the driver must support that runtime.
- Run `uv pip check --python .venv/bin/python` after repairing or deliberately changing the managed environment.
- Re-run `./install.sh --accelerator nvidia --repair` (or `.\install.ps1 -Accelerator nvidia -Repair`) to restore the reviewed NVIDIA profile; it cannot repair a missing or incompatible driver.
- A failed CUDA probe intentionally falls back to CPU. Read the recorded probe errors instead of assuming the fallback is accelerator-backed.

If CUDA reports an illegal memory access or poisoned context, clearing the cache may not be enough. Stop work and restart the backend process before retrying with a safer resource plan.

The normal `./run.sh` or `.\run.ps1` entrypoint keeps a lightweight supervisor outside
the model-owning worker. Stop first requests cooperative cancellation and
removes queued runs. If a third-party model call does not return within the
bounded grace period, the worker is replaced so the operating system releases
its RAM/VRAM before another run is accepted. If a native model runtime exits
unexpectedly, the supervisor records the active run as failed, cancels queued
in-memory work that cannot be reconstructed safely, and starts a clean worker.
The browser reconnects automatically; the interrupted run does not resume from
its last denoising step and must be retried with a resource plan that fits.

## Open-file exhaustion during model loading or generation

`file_descriptor_limit` and `huggingface_file_descriptor_limit` identify a
process or system file-descriptor limit, including native errors hidden by a
subsequent model-loader decoding error. Inspect the model-owning worker's
`/proc/<PID>/limits` and descriptor count on Linux. A systemd service may inherit
a soft limit of only 1,024 even when the shell or hard limit is much larger.

ROCm expandable allocations can retain one descriptor per allocation segment.
A constrained worker can therefore report HIP out-of-memory while substantial
VRAM remains free. Confirm descriptor growth and release before concluding that
the model is too large or that its weights are corrupt. Increase the service's
soft `LimitNOFILE` within its existing hard limit, restart only while the queue
is idle, and retry the original recipe while monitoring descriptors. Record this
as a deployment change separately from application code. Do not increase limits
indefinitely if descriptors continue growing after models and caches are released.

After a failed model download, an empty active-download list does not prove the
model is installed. Require Model Manager's exact reviewed artifact to report
Ready before retrying generation; partial cache files are preserved for repair.

## Video export completed but its file is missing

A completed graph receipt alone does not prove that a video was encoded. Verify
that its media URL returns a non-empty file and that the file decodes. The Video
Export node accepts NumPy video batches in `B,F,H,W,C` layout, including the
pinned Helios decoder's default `output_type=np`, and exports each batch item as
its own clip. Empty or unsupported inputs and missing encoded files must fail
instead of returning a preview URL for a nonexistent artifact. Keep the original
decoder output type when reproducing export failures.

## Apple MPS or Intel XPU is unavailable

Install or repair the Apple profile:

```bash
./install.sh --accelerator mps --repair
```

Preflight should list an `mps` device when both the host and PyTorch build support it. Current MPS support is not CUDA parity. Large Qwen Image, Wan Video, quantized CUDA, Nunchaku, xformers, FlashAttention, and SageAttention paths may be unavailable or impractical. A contract test or visible MPS device is not evidence that a large model completed on Apple hardware.

For supported Intel graphics on x86-64 Linux or Windows, install or repair the preview profile with `./install.sh --accelerator intel --repair` or `.\install.ps1 -Accelerator intel -Repair`. Preflight must report `xpu:0` and pass a real XPU tensor. Integrated Intel graphics share system memory; MoDiff does not enable CUDA-only CPU-offload hooks on XPU. A visible device makes the path available, but only an exact model/recipe receipt qualifies it.

## A model is missing, gated, or repeatedly downloads

- Open model/cache diagnostics and confirm the selected repository and revision are complete.
- Check `[huggingface] online_status`; `Offline` prevents missing files from being fetched.
- For gated repositories, accept the model's terms on Hugging Face and configure a read token.
- The token saved through the UI is plaintext in ignored `config.ini`. Do not paste it into logs or issues.
- Set `[huggingface] cache_dir`, `HF_HOME`, or `HF_HUB_CACHE` to a writable volume with sufficient free space. A configured `cache_dir` is exported as `HF_HUB_CACHE` by the backend.
- Avoid pointing multiple applications at partially compatible cache layouts unless you understand how snapshots and revisions are resolved.

Opening or refreshing Model Manager can require a full index and artifact
validation pass over a very large cache. Those scans run in background worker
threads and simultaneous refresh requests share the same work, so health,
workflow, and queue requests should remain responsive. The UI may continue to
show the previous screen while the fresh generation is being validated, but a
failed fresh validation is reported as an error; MoDiff does not substitute
older cached readiness results. If unrelated API calls time out while this scan
is active, confirm that the backend process includes the current discovery
implementation before treating the model worker as disconnected.

The same refresh also reads runtime status, optional runtimes, and the Studio
model-capability catalog. Their hardware/package probes, filesystem inspection,
catalog construction, and JSON encoding are likewise off the request loop and
coalesced. The default multi-megabyte capability response is pre-serialized
before the HTTP listener opens. `/health` uses the stable runtime identity
captured during backend startup; changing the managed runtime requires the
documented backend restart. Dynamic RAM, disk, and accelerator statistics
remain available from `/system_stats` without blocking other requests.

Current app-owned model and Gallery snapshot payloads use standard Hub HTTP,
which retains bounded per-request timeouts and retries while preserving already
completed cache blobs. Progress is still derived partly from cache
materialization and may remain indeterminate or jump. Completion requires the
backend's validation result, not merely network inactivity or a `100%` UI
estimate. A worker already running during an app update keeps its original
download behavior until it is restarted; do not interrupt an active transfer
solely to switch transports.

## A workflow runs out of memory

1. Inspect the structured run failure and current GPU-process report.
2. Use an Auto resource plan or explicitly choose a lower-memory model/quantization/offload mode.
3. Close unrelated accelerator-heavy applications.
4. Request best-effort cleanup through the UI or `POST /runtime/gpu_cleanup`.
5. Restart the backend if the CUDA context is unhealthy.

Cleanup can release MoDiff's node cache, managed Diffusers components, memory-manager entries, and accelerator cache. It cannot free memory owned by another process, and it does not guarantee that the same workflow fits afterward.

## A run is taking much longer than expected

### Resource monitoring during execution

Allocator statistics can show as paused while a task is active. This is deliberate:
Torch allocator probes can hold the GIL during inference even when called from a
background thread. The monitor retains device identity and available OS samples,
and shows unavailable counters as unknown rather than zero or stale values.
Idle allocator sampling resumes after execution; final run measurements remain
separate. See the [resource API](api-reference.md) for the response fields.

Use WebSocket progress and the lightweight `/queue` snapshot during inference;
request full `/runs/{task_id}` details for terminal provenance or explicit
inspection. Concurrent resource and history reads are coalesced, and history
serialization and mutations run off the HTTP event loop with ordered writes.
These boundaries reduce observer-induced stalls but do not guarantee a latency
bound for every model or native library operation. If unrelated requests still
time out, retain bounded health/queue/resource samples and profile the actual
running source. Distinguish a slow worker from a dead worker using supervisor
status. Do not delete history, weaken timeout assertions, or reduce generation
settings to conceal the problem.

### Inference and recovery

First distinguish slow progress from a stalled worker. A step counter that
continues to advance, an active node/phase in Queue, websocket heartbeats, or
sustained accelerator activity indicates that the run is alive even when the
ETA is long. A static counter with no task update, heartbeat, log activity, or
resource activity for an extended period needs investigation.

The visible phase matters:

- **Download/validation** can wait on repository metadata, network transfer,
  hashing, or cache materialization.
- **Loading/placement** can move tens of gigabytes of weights before the first
  denoising step. Shared system memory is much slower than equivalent-capacity
  discrete VRAM and may make a technically runnable plan impractical.
- **Denoising** normally repeats similar work for every requested step; use the
  observed seconds per completed step for a rough remaining-time estimate.
- **Decode/export** can be expensive for high-resolution images or long video
  and audio outputs and may depend on local FFmpeg performance.

Auto chooses a qualified runnable recipe, not a guaranteed performance tier.
Without changing the prompt or generation parameters, a later run may be
faster with a qualified pre-quantized artifact, supported attention backend,
compile/cache option, or improved device placement. Those changes require the
pipeline to be loaded again and cannot safely accelerate an active run. Do not
enable an unqualified quantizer or optional kernel as a recovery experiment.

For a first smoke test, choose a lightweight Auto-ready image recipe. If input
settings may be changed, lower resolution, steps, frames, or duration before
testing large final settings. If progress is still advancing, stopping is a
user tradeoff rather than a crash-recovery requirement.

## Video or audio export fails

- Install FFmpeg and ensure it is available on `PATH`.
- Confirm `imageio` and `imageio-ffmpeg` are installed in the active environment.
- Check that `[paths] videos`, `[paths] images`, and `[paths] temp` are writable.
- Verify input media is supported by the local FFmpeg build.
- On Windows, restart the shell after changing `PATH`.

## Files are not visible or writes fail

The file browser is scoped to `[paths] work_dir`; relative paths resolve below it. Confirm the file is inside that boundary and the backend process can read it. Runtime persistence and media directories also require write access.

Do not fix a permission error by pointing `work_dir` at an entire home directory or disk root. Create a dedicated MoDiff workspace with the narrow permissions required by your workflows.

## Custom modules do not appear

- Confirm the module is under `custom/<Name>/` and has an importable package structure.
- Review its dependencies and imports before enabling it.
- Restart or use the registry refresh after changes.
- Inspect the console for the first import failure; one optional module should not be allowed to hide failures in another.
- Remember that installing or refreshing a module imports trusted Python code with backend-process permissions.

Built-in module contributors should follow [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-node-module).

## Safe local cleanup

These are disposable when the backend and tests are stopped:

- `.pytest_cache/`, `.ruff_cache/`, and `__pycache__/` directories.
- Repository-local log and smoke-test artifacts.
- A local `.venv/`, if you are prepared to recreate it with `./install.sh --accelerator auto --repair` or the equivalent Windows installer command.

Do not broadly delete `data/`, Hugging Face caches, or `config.ini` while troubleshooting. They may contain models, generated media, prompts, blocks, workflow shares, planner history, or tokens. Review specific paths and back up anything important first.

## Mochi fails before component placement

The reviewed Mochi loader enables mandatory tiling through the pinned
Diffusers API, `pipeline.vae.enable_tiling()`, after applying the execution
recipe and before offload placement. The pipeline itself has no
`enable_vae_tiling()` method. A loader calling that pipeline-level method is
an adapter defect; changing drivers, the prompt, or the generation settings
will not repair it. The indexed T5 selection, BF16 variant, and original
offload policy remain part of the reviewed route.

## GLM-Image rejects unequal prompt embedding shapes

GLM glyph encoding can produce a long positive sequence and a one-token empty
negative sequence. The pinned pipeline supports those lengths during native
encoding, but rejects unequal embeddings supplied directly to its public call.
MoDiff keeps GLM's native validation, prior generation and encoding order. A
call-scoped dtype adapter requests the diffusion transformer's dtype from
`encode_prompt`, while the text encoder stays float32. The original method is
restored on success or failure. Prompts and embeddings are neither truncated
nor padded to satisfy the public precomputed-embedding check.


### Missing Modular conditional companion snapshot

If a reviewed block reports `Cannot load conditional snapshot`, restore
`data/modular-conditional-contracts.json` using
`scripts/generate_modular_conditional_contracts.py` in the pinned optional
Diffusers runtime. This is a no-weight structural generator. Its coverage comes
from the validated reviewed workflow snapshot, preserving constructor configs;
routing registry promotions must not remove classes from this companion.
The generator validates hashes, branch truth tables and execution traces against
the existing resolved snapshots. Run its `--check` mode and the conditional
contract tests after restoration. Do not substitute an empty snapshot or bypass
validation. This generated deployment file is distinct from curated media review
records, which cannot be reconstructed by inventing approvals.
