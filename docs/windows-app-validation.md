# Windows app validation plan

This plans future Windows tests; Windows GPU execution and fit remain unrun.
Native install/preflight/model-free smokes have CPU CI coverage. Check the final
pushed commits and new Windows CI result after the source-inventory correction;
the local full backend pass and ROCm images do not qualify Windows NVIDIA.
Use [three-machine validation](three-machine-app-validation.md) for shared
ownership/evidence procedures, [developer setup](developer-setup.md) for dependencies,
and [image validation](image-template-validation.md) for exact recipes/contracts.
Use the existing app/executor/resource planner, without qualification edits.

## Scope and original feedback acceptance

Baseline: Windows x86-64/NVIDIA, reviewed CUDA, eager/native SDPA, production
client. Record actual hardware; test dedicated 16 GB VRAM/32 GB RAM explicitly
if available. Windows AMD is installation-blocked pending reviewed SDK/wheels;
Intel XPU preview and optional compilation are separate targets. See the
[support matrix](runtime-support-matrix.md). Keep original feedback numbering
and independent pass/fail/blocked/unrun results.

| Feedback                                                                        | Observable test and pass criterion                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Failure evidence                                                                                                                                          |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. The advertised no-script setup still calls scripts                           | Complete the native `uv sync` / `uv run` and client `npm ci` path below without `install.ps1`, `run.ps1`, or a project shell wrapper. Test the guided installer separately. Verification scripts are identified as tests, not native installation prerequisites.                                                                                                                                                                                                                         | Exact commands, native exit codes, installer invocation if unexpectedly required, and the point the instructions diverge from the app.                    |
| 2. Redundant `--no-sync` / `--no-project` flags                                 | Ordinary native setup and launch succeed with the documented same-extra commands, without either flag. Direct venv Python is used when intentionally retaining test-only packages installed with `uv pip`.                                                                                                                                                                                                                                                                               | Command and resolver log; missing package or unexpected environment replacement.                                                                          |
| 3. Windows expandable segments and missing Triton                               | Before importing Torch, the default allocator policy is `torch_default` with no injected allocator variable. Original eager image recipes run without a Triton/compiler activation prerequisite. Test an actual compiled GPU kernel separately if requested.                                                                                                                                                                                                                             | Allocator policy and environment-variable presence, preflight/runtime package identities, full warning/error, eager task result, separate compiler probe. |
| 4. Transformers is required for mainstream Diffusers                            | Fresh base installation imports and verifies Transformers; ordinary supported image loading does not offer an optional Transformers Install/Activate prerequisite.                                                                                                                                                                                                                                                                                                                       | Base package/version/origin, `optionalRuntimeRequirement`, visible blocker and task/load receipt.                                                         |
| 5. PEFT is required for LoRAs                                                   | Fresh base installation imports and verifies PEFT. A real immutable LoRA loads, affects output, can be replaced, and restores the declared adapter set without optional core activation.                                                                                                                                                                                                                                                                                                 | Exact adapter bytes/names/scales/order, successful load/use observations, outputs and replacement failure.                                                |
| 6. Why pin uv, and how to upgrade                                               | Record the operator's uv version; there is no exact uv executable version requirement. Use `uv self update` for standalone uv or its package manager's upgrade command. Keep app lockfile and reviewed runtime pins distinct from uv itself.                                                                                                                                                                                                                                             | Tool versions and resolver/preflight result before/after; do not silently upgrade the model runtime during a comparison.                                  |
| 7. `Error: write ECONNABORTED` on every refresh/start                           | Production and Vite sessions each survive the bounded startup/refresh checks below. A same-task reconnect recovers progress without a duplicate submission. Expected disconnected requests do not cause repeated unhandled errors; genuine backend failures remain visible.                                                                                                                                                                                                              | Browser console, Vite stdout/stderr, failed URL/status, WS close/reconnect, health/instance and Queue timeline.                                           |
| 8. Missing Transformers/PEFT immediately produces a UI error                    | On the correct fresh install, both are available and Setup reports base readiness before a supported template runs. A deliberately incomplete disposable CPU test environment must show a bounded actionable dependency error, not a blank app or unexplained Activate loop. Never remove packages from the working CUDA environment.                                                                                                                                                    | Preflight, Setup screenshot, package status, server error and recovery after same-extra sync/restart.                                                     |
| 9. Optional install has unclear completion, activation/reload takes a long time | For an actually actionable optional profile, the same durable job shows install/validation/ready and activation/restarting/verifying/terminal progress across refresh. Completion requires the replacement worker's verified exact environment. A temporary disconnect retains progress; it must not reset misleadingly to Activate. Core Transformers/PEFT need none of these actions.                                                                                                  | Catalog/action availability, exact profile digest/environment/job IDs, timestamped job states, old/new worker and health, failure/recovery hint.          |
| 10. First image succeeds; repeated Auto rejects host memory                     | Run the unchanged original cold/warm/changed-input/model-switch sequence. Reconcile each admission with contemporaneous free host RAM, dedicated VRAM, owned live caches and declared reclaim/reuse. Preserve the original message: `Combined systemRamBytes: needs 8589934592 bytes; 4279840768 bytes available`. These are host RAM bytes, not VRAM. A real shortage is a valid bounded rejection; stale accounting, duplicate owners or failure to credit eligible reuse is a defect. | Full Auto plan/blocker, runtime fingerprint, resource receipt, actual process/host/device samples and previous completed task/model-owner identities.     |

The original memory report did not establish OS/model/free RAM. The overall
target is supported templates working through normal Auto with honest readiness
and no hidden recipe substitutions.

## 1. Freeze source and create private storage

Run PowerShell without a profile. Use paired reviewed pushed SHAs, a fresh root
per native/guided/optional case, and no copied environments, tokens, configs,
private workflows, approvals, overlays or journals.

```powershell
$modiffBackendSha = '<FINAL_PUSHED_BACKEND_SHA>'
$modiffClientSha = '<MATCHING_PUSHED_CLIENT_SHA>'
$modiffTestRoot = Join-Path $env:LOCALAPPDATA ('MoDiff-validation-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $modiffTestRoot -ErrorAction Stop | Out-Null
$modiffBackend = Join-Path $modiffTestRoot 'MoDiff'
$modiffClient = Join-Path $modiffTestRoot 'MoDiff-client'
$modiffRunRoot = Join-Path $modiffTestRoot 'run'
$modiffEvidence = Join-Path $modiffTestRoot 'evidence'
New-Item -ItemType Directory -Path $modiffRunRoot, $modiffEvidence | Out-Null
git clone https://github.com/sdevil7th/MoDiff.git $modiffBackend
if ($LASTEXITCODE -ne 0) { throw 'Backend clone failed' }
git clone https://github.com/sdevil7th/MoDiff-client.git $modiffClient
if ($LASTEXITCODE -ne 0) { throw 'Client clone failed' }
git -C $modiffBackend checkout --detach $modiffBackendSha
if ($LASTEXITCODE -ne 0) { throw 'Backend checkout failed' }
git -C $modiffClient checkout --detach $modiffClientSha
if ($LASTEXITCODE -ne 0) { throw 'Client checkout failed' }
git -C $modiffBackend rev-parse HEAD
git -C $modiffClient rev-parse HEAD
```

Before imports, write checkout-local `config.ini`: there is no
`MODIFF_CONFIG_PATH` or `main.py --config`. `MODIFF_MANAGED_ROOT` alone is
insufficient. Keep `app_root`/tracked curated source in the checkout; redirect
all eight mutable paths and the Hub cache:

```powershell
$modiffStorageKeys = 'work_dir','data','images','videos','audio','models','upscalers','temp'
$modiffPathLines = foreach ($modiffKey in $modiffStorageKeys) {
    $modiffDirectory = Join-Path $modiffRunRoot $modiffKey
    New-Item -ItemType Directory -Path $modiffDirectory | Out-Null
    "$modiffKey = $modiffDirectory"
}
$modiffHubCache = Join-Path $modiffRunRoot 'hub'
New-Item -ItemType Directory -Path $modiffHubCache | Out-Null
$modiffIni = @"
[server]
host = 127.0.0.1
port = 8088
[huggingface]
cache_dir = $modiffHubCache
online_status = Auto
[paths]
$($modiffPathLines -join "`n")
"@
[IO.File]::WriteAllText((Join-Path $modiffBackend 'config.ini'), $modiffIni,
    [Text.UTF8Encoding]::new($false))
$env:MODIFF_MANAGED_ROOT = Join-Path $modiffRunRoot 'managed'
$env:HF_HOME = Join-Path $modiffRunRoot 'hf-home'
$env:HF_HUB_CACHE = $modiffHubCache
$env:TORCH_HOME = Join-Path $modiffRunRoot 'torch'
$env:TEMP = Join-Path $modiffRunRoot 'temp'
$env:TMP = $env:TEMP
```

These are shell-local assignments, not user/machine changes. Never assign
`$PID`/`$HOME`, log auth values, or stop another port owner. Check port 8088 and
supervisor availability; inspect preflight's port observation even on exit 0.
Record OS/build, CPU, total/free RAM/disk, GPU dedicated/shared memory, driver,
WDDM/display usage, tool/browser versions and background workloads using System
Information/Task Manager/available `nvidia-smi`. Budget exact weight/cache/temp
space; do not change operator processes or settings.

## 2. Prove native and guided installation independently

Native setup uses the Node/npm versions in developer setup and current lockfiles:

```powershell
Set-Location $modiffBackend
uv --version
uv sync --extra cuda *> (Join-Path $modiffEvidence 'native-sync.log')
if ($LASTEXITCODE -ne 0) { throw 'Native CUDA sync failed' }
$modiffPython = Join-Path $modiffBackend '.venv\Scripts\python.exe'
uv pip check --python $modiffPython
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
uv run --extra cuda python -m modiff.preflight --json --check-port 8088 --fail-on-error `
    *> (Join-Path $modiffEvidence 'native-preflight.log')
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; inspect its report' }
Set-Location $modiffClient
node --version
npm --version
npm ci *> (Join-Path $modiffEvidence 'native-npm-ci.log')
if ($LASTEXITCODE -ne 0) { throw 'Client dependency install failed' }
```

Require Python 3.12, reviewed Diffusers source, compatible Transformers/PEFT
and CUDA readiness. Preserve stderr/exit codes; progress on stderr is not failure.
While idle, run the existing no-weight CPU interoperability/LoRA and temporary-
server smokes; neither proves full GPU generation:

```powershell
Set-Location $modiffBackend
& $modiffPython scripts\smoke_base_runtime.py *> (Join-Path $modiffEvidence 'base-smoke.log')
if ($LASTEXITCODE -ne 0) { throw 'Base interoperability smoke failed' }
& $modiffPython scripts\smoke_service_package.py *> (Join-Path $modiffEvidence 'service-smoke.log')
if ($LASTEXITCODE -ne 0) { throw 'Service smoke failed' }
```

Contributor tests use a separate private CPU checkout: `uv sync --extra cpu`,
`uv pip install --python .venv\Scripts\python.exe -r requirements\test.txt`, then
`.\.venv\Scripts\python.exe -m pytest -q`. Direct Python retains test-only packages
that `uv run` can re-sync away. Never CPU-sync over CUDA.

Repeat the preparation in a second fresh paired checkout for the guided path:

```powershell
.\install.ps1 -Accelerator nvidia -SystemCheck -Json *> guided-system-check.log
if ($LASTEXITCODE -ne 0) { throw 'Guided system check failed' }
.\install.ps1 -Accelerator nvidia -NonInteractive *> guided-install.log
if ($LASTEXITCODE -ne 0) { throw 'Inspect required actions or installer failure' }
.\run.ps1
```

Require real CUDA tensor/profile validation, client build and readiness.
Preserve structured required actions; no bypass. `-Resume`/repair are separate
recovery attempts, not fresh-install passes. There is no `-Guide` parameter.
Follow developer setup for uv/direct-pip upgrades; runtime changes need new gates.

## 3. Verify default runtime and served application identity

With allocator variables absent in this disposable shell, verify before Torch
imports. Inherited settings may be cleared here only, never machine-wide:

```powershell
Set-Location $modiffBackend
& $modiffPython -c "import json,sys; from modiff.runtime_environment import configure_allocator; p=configure_allocator(); assert 'torch' not in sys.modules; assert p['source']=='torch_default' and p['setting'] is None; print(json.dumps(p))"
if ($LASTEXITCODE -ne 0) { throw 'Default pre-Torch allocator check failed' }
uv run --extra cuda python main.py
```

Capture app stdout/stderr. Each new terminal must restore these variables using
the exact root created above (do not create another root), then `Set-Location`.
The same preparation applies before later sampler, Vite and compiler commands:

```powershell
$modiffTestRoot = '<EXACT_EXISTING_TEST_ROOT>'
$modiffBackend = Join-Path $modiffTestRoot 'MoDiff'
$modiffClient = Join-Path $modiffTestRoot 'MoDiff-client'
$modiffRunRoot = Join-Path $modiffTestRoot 'run'
$modiffEvidence = Join-Path $modiffTestRoot 'evidence'
$modiffPython = Join-Path $modiffBackend '.venv\Scripts\python.exe'
$env:MODIFF_MANAGED_ROOT = Join-Path $modiffRunRoot 'managed'
$env:HF_HOME = Join-Path $modiffRunRoot 'hf-home'
$env:HF_HUB_CACHE = Join-Path $modiffRunRoot 'hub'
$env:TORCH_HOME = Join-Path $modiffRunRoot 'torch'
$env:TEMP = Join-Path $modiffRunRoot 'temp'
$env:TMP = $env:TEMP
Set-Location $modiffBackend
```

Retain `/health`, `/runtime/status`, `/nodes`, `/model_capabilities`,
`/runtime/optional-runtimes`, `/runtime/resources` and `/queue`: verify readiness,
package versions/profile/source/instance/cwd/paths. Health does not expose package
import origins; capture those separately from the idle private runtime if needed.

```powershell
$modiffOrigin = 'http://127.0.0.1:8088'
$modiffHealth = Invoke-RestMethod "$modiffOrigin/health"
$modiffWorkerProcessId = (Get-NetTCPConnection -State Listen -LocalPort 8088 |
    Where-Object LocalAddress -eq '127.0.0.1').OwningProcess
if (@($modiffWorkerProcessId).Count -ne 1) { throw 'Expected one loopback worker listener' }
Get-CimInstance Win32_Process -Filter "ProcessId = $modiffWorkerProcessId" |
    Select-Object ProcessId, ParentProcessId, CreationDate, ExecutablePath, CommandLine
$modiffHealth | ConvertTo-Json -Depth 100
if ($modiffHealth.workerControl.available) {
    Invoke-RestMethod ($modiffHealth.workerControl.address + '/health')
}
```

Match supervisor `workerPid` to listener PID/creation/cmdline and health cwd/instance;
Windows uv/venv redirectors can be different processes. Never guess the PID or
control port. For production, hash fetched `/` and its actual asset URLs against
`web`, binding the build's client SHA. Separately test Vite with matching source:

```powershell
Set-Location $modiffClient
$env:VITE_BACKEND_PROXY_TARGET = 'http://127.0.0.1:8088'
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Record Vite source/proxy/HMR separately and freeze source during comparisons.
Client `npm run check` / `check:ui` are separate gates; mocked transport is not
actual GPU/browser evidence. Do not manually replace assets.

## 4. Startup, editing and model-free app checks

For production and Vite independently: five fresh-document launches and twenty
normal/hard refreshes. Retain UTC health/instance, HTTP/WS and console/Vite logs.
Require Gallery/Nodes/Setup/Queue readiness, no fresh-document 404, bounded
restart recovery and no recurring unhandled abort errors. Genuine down/timeout/
500/malformed responses must remain visible; do not blanket-filter disconnects.

Use new workflows and the common guide's UI procedures. Minimum acceptance:

- Manual/Auto text graph: correct value, matching terminal Queue and `/runs/{task_id}`.
- Add/search/new stage controls, committed field edits, click/drag wiring,
  delete/reconnect, Undo/Redo, save/reopen/refresh: exported values/wires preserved.
- Group/expand/move/resize/ungroup/regroup: working sockets, immutable definition
  and customization retained. Bundle/raw imports keep owners/Guider/required inputs.
- Approved image/audio/video decoding/playback/export: correct orientation,
  dimensions/duration/sample rate; no claim of model generation from playback.
- Explicit reviewed custom Add/Load, source edit/Reload with new identity,
  Disable/reopen and bounded missing-node error; import never approves code.
- Text service export and existing inspect/build/run CLI: named outputs and
  custom drift rejection. Commands: [service prototyping](service-prototyping.md).

## 5. Acquire models and preserve full recipes

Check exact model/task/profile/access against current capabilities. Download
immutable base/auxiliary/LoRA revisions through Models into the private cache;
verify Gallery reference hashes and actual loader resolution. Keep file/job
progress and completion. Gated access/terms must be satisfied by the tester.
Retain 401/403, revision/disk/weight failures and separately scoped retries;
never bypass access or substitute unpinned/repository-code models. Redact secrets.

Export visible/API graphs and preserve the image guide's complete consumed
recipe: positive/negative, integer seed/generator, steps/geometry, guidance,
scheduler, dtype/quantization/offload, attention/VAE/adapters, ordered input/mask
bytes, crop/strength/canvas/stitch. No shorter recipe or hidden policy substitution.
Select exact supported profiles; this matrix gives priority examples, not full
coverage of all templates or all admitted models. Use the matching client
[current template inventory](https://github.com/sdevil7th/MoDiff-client/blob/fix-ui-ux-issues/docs/image-template-workflows.md#current-template-inventory)
and live capability/task contracts to enumerate every intended test, including
other admitted SDXL/SD3 or media routes. Each untested row stays explicitly unrun:

| Route                                                                | Required distinct check                                                                                                                                                                                                                  |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Z-Image native text-to-image                                         | Full original 8-step recipe; include a nonempty negative, genuine re-encoding after prompt changes, and full LoRA recipe when available.                                                                                                 |
| Qwen Image text-to-image                                             | Full original 50-step recipe; same original input/settings through raw wiring and opt-in component bundle, with genuine generation in both.                                                                                              |
| Qwen Edit / Edit Plus                                                | Original single/multiple reference recipes and their full 4/40/50-step settings; retain ordered references and actual effective placement.                                                                                               |
| Qwen object / mask / two outpaint templates                          | Original full 8/16/12/20-step native compatibility recipes; verify owner/encoder policy, mask composite, strength, authored dimensions/crop and canvas edges. Generic developer Edit defaults remain separate.                           |
| Qwen Layered                                                         | All original local output items, RGBA/alpha and layer order; a single attractive flattened image is insufficient.                                                                                                                        |
| Qwen ControlNet                                                      | Both original 36-step/768² recipes with scales 1.2 and 1.1, exact auxiliary weight/input and cold/warm runs. Preserve earlier genuine failures; successful current output is functional coverage when no same-machine old output exists. |
| Native Flux Dev/Krea/Schnell/Kontext                                 | Exact full original steps and guidance contract; Kontext single/multiple references and actual stitching need separate checks.                                                                                                           |
| Flux Fill / Fill outpaint / Canny / Depth / Redux edit / Redux multi | Six legitimate whole operations at the reviewed pin. Run their original full recipes; do not replace concatenated conditioning/prior semantics with decorative native stages or an unrelated ControlNet.                                 |
| Upscale                                                              | Exact immutable Spandrel weights/tile/downscale settings and both generation/upscaled Preview sinks, preserving receipt/item order.                                                                                                      |
| Audio/video/other supported model tasks                              | Separate current task-specific runtime/admission, license and full model/input/output checks from model-free media playback. Record untested routes explicitly rather than claiming image coverage qualifies the whole app.              |

## 6. Repeated Auto, memory, LoRA and active user journey

Serialize owned GPU tasks. Capture idle health/Queue/free RAM/dedicated VRAM,
Auto plan/effective mode/owners/fingerprint before allocation. Preserve a rejected
canonical recipe; a separately labeled supported recipe can continue UI checks.
Run A-E without explicit release/restart:

| Run        | Change                                          | Required observation                                                                                                                                                  |
| ---------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A: cold    | Original recipe, no previously loaded owner     | Accepted ordinary Auto, successful load and complete original output set.                                                                                             |
| B: warm    | Same saved recipe                               | Completed matching task; identify genuine generation versus delivered cached output from actual node/task observations. Do not invent model forwards for a cache hit. |
| C          | Seed only                                       | Saved numeric seed and consumed seed match; a genuine generation and different random input/output where measured.                                                    |
| D          | Positive prompt and seed                        | Negative/settings retained; actual new encoding and generation, not reused incompatible embeddings or a stuck completed CFG loop.                                     |
| E: restore | Original prompt/seed/settings                   | Actual restored execution or honestly labeled cache reuse; compare complete original outputs and contracts.                                                           |
| F then G   | Switch to a second supported model, then return | Honest plan/owner replacement/reuse/reclaim; both admitted runs finish or give an accurate bounded capacity blocker. No stale guider/adapter/encoding state.          |

For dedicated 16 GB VRAM/32 GB RAM, require A-E per intended qualification recipe.
Retain separate `systemRamBytes`/`vramBytes`, free capacity/reserve/reuse/reclaim;
`model_cpu` is not physical-residency proof. No unrelated cleanup/artificial
pressure/recipe reduction. Sample the verified worker in a separate terminal:

```powershell
& $modiffPython scripts\capture_runtime_memory.py --pid $modiffWorkerProcessId `
    --output (Join-Path $modiffEvidence 'host-memory-part1.jsonl') `
    --duration-seconds 3600 --interval-seconds 0.1 --full-memory-interval-seconds 1
```

Require header/first sample before Run and one writer; rotate only idle, keeping
footer/header/gap. PID creation is bound internally, not a CLI option. RSS/USS
are sampled lower bounds; Windows PSS may be absent. Preserve terminal allocator
peak version/source and unknown missing peaks. Live `paused_during_execution`/
null counters are intentional: use Queue/OS then terminal receipts, no competing
Torch probe. Total capacity or telemetry alone does not qualify fit.

LoRA: exact base/adapter revision/SHA/name/order/scale/replace semantics. Compare
base, nonzero scale, changed scale, second-adapter replacement and restored set.
Require actual load/activation/use where observed plus visual/output effect;
no stale/duplicate adapters after replacement. A descriptor alone is insufficient.

During genuine generation, five active refreshes must recover the same unique
task/progress/output. Check owning Studio restoration separately from a fresh
Queue observer, which need not become owner. An unrelated edited tab must stay
intact. Save/reopen preserves values/wires and correct task/source output.
On a separate owned run, ordinary Stop must reach a terminal reason and drained
queue before full-recipe Retry/new task. Test pending removal separately.
Record graceful cancel versus emergency replacement, new PID/instance/readiness;
never retry an unknown active task. Follow the common guide's recovery procedure.

## 7. Optional runtime and compiler checks are separate

Core Transformers/PEFT are base-delivered: never uninstall them to follow
historical overlay campaigns. Additional profiles must be qualified/cutover-ready
with an offered action; unavailable actions stay closed. In a separate managed
case use normal Setup Install then explicit Activate/returned environment.
Retain `/runtime/optional-runtimes` and `/runtime/optional-runtimes/jobs/{job_id}`
through refresh/restart: same durable job, install/validation/ready and activation/
restart/verification/terminal. Require replacement origins/environment/digest/
readiness, not HTTP 202 or a directory. Test offered install cancellation,
failure/rollback and ordinary task afterward; serialize runtime mutation.
See [optional lifecycle](optional-runtime-optimizations.md).

Compiler install needs separate reviewed PyTorch-compatible Windows packages;
there is no invented `triton-windows` pin/install command here. At idle after restart:

```powershell
$modiffCompileResponse = Invoke-RestMethod -Method Post -Uri "$modiffOrigin/runtime/optimizations/probe" `
    -ContentType 'application/json' -Body '{"capabilityId":"regional_compile"}'
$modiffCompileResponse | ConvertTo-Json -Depth 100
```

Require `receipt.status = probe_passed`, `receipt.validationStatus = passed`,
the expected `capabilityId` and matching `runtimeFingerprintHash`. The public
receipt exposes no `result`, internal `detail.executed`, device or backend.
The existing backend probe checks actual execution, but can select CPU when CUDA
is unavailable. To prove the exact GPU (and, separately, required Flex), run the
existing bounded helper at idle, without weights:

```powershell
& $modiffPython -c "import json; from modiff.runtime_compilation import require_compilation; r=require_compilation(device='cuda:0',require_flex=False); assert r['available'] and r['device']=='cuda:0'; print(json.dumps(r))"
if ($LASTEXITCODE -ne 0) { throw 'Actual CUDA compiler probe failed' }
```

For a Flex-required route, repeat with `require_flex=True` and require the returned
`flex_executed = true`. CPU/import success is insufficient. Failure leaves eager
usable; required compilation blocks before weights. Auto optimization also needs
an exact baseline, improvement, output review and current receipt.

## 8. Same-hardware comparison and bounded handoff

Windows parity requires actual original/candidate on this machine with pinned
runtime/models/input bytes and separate app identities. An old failure means
current functionality, not parity. Keep all ordered sinks/items, raw SHA-256 and
decoded dimensions/mode/alpha. Use the image guide's consumed contracts/tool:

```powershell
& $modiffPython scripts\compare_image_template_outputs.py `
    --baseline <OLD_PNG_PATHS_IN_OUTPUT_ORDER> --candidate <NEW_PNG_PATHS_IN_OUTPUT_ORDER> `
    --baseline-contract <OLD_CONSUMED_CONTRACT_JSON> `
    --candidate-contract <NEW_CONSUMED_CONTRACT_JSON> --report <NEW_COMPARISON_JSON>
```

Replace placeholders; exact decoded pixels are default, no tolerance widening.
PNG metadata may differ. Preserve requested-interface/offload differences and
missing observations. Equal pixels do not prove RNG/work/residency/fit/quality;
review prompt, edges/text/LoRA/artifacts separately.

Timebox preparation/startup to an initial 30-60-minute diagnostic checkpoint;
downloads/install can exceed it and remain prerequisites, not assumed passes.
Then prioritize A-E/resource/refresh/cancel, LoRA and exact input/multiple-output
routes, followed by remaining families/optional/compiler. Recover the original
failing model identity or label it unknown. Batch families without substitutions;
measure full durations, never scale tiny tests. At window end, finish an active
full run honestly and start no unbounded new task or shortened substitute.

Each failure needs action/UTC/source/runtime/model, visible/API graph, redacted
request/response, task/job/instance, receipt/Queue/health/logs, plan/samples and
all outputs. Classify dependency/access/input/runtime/capacity/accounting/
reconnect/numerical/quality; HTTP success alone is not model success.
At known idle stop only owned app/sampler/Vite, retain footer/port/process absence
and preservation receipts. Report each pass/fail/blocked/unrun, duration/hash and
private evidence path. No operator cleanup, secrets/private history in Git, or
qualification promotion from this plan.
