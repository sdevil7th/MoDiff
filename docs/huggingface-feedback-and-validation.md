# Hugging Face feedback: implementation, validation and remaining work

Reviewed on **9 October 2026**. This document is the shared handoff for the
Hugging Face installation/UX feedback, the simpler image-node rework and the
Linux/AMD work to compare with Windows changes. It records the planned behavior,
implemented approach, expected acceptance, existing results and open fixes.

The overall expectation was straightforward: complete normal setup, choose an
official template and generate, then generate again or edit the workflow without
having to install foundational libraries or diagnose memory placement manually.
Model downloads and gated access can require explicit action; their status must
be understandable before generation starts.

## Source and evidence scope

| Source                             | Reviewed checkpoint                                                                                                                                         | Evidence boundary                                                                                                                            |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Published backend                  | `5f38a40af6d9f3befbd4c744a4700ab9865e323d`, branch `fix-ui-ux-issues`                                                                                       | Clean, pushed; full local CPU gate and completed cross-platform CPU CI                                                                       |
| Published client                   | `bfad8134b82f6b77affbd10599a71740c578cae1`, same branch                                                                                                     | Clean, pushed; full client quality/browser gates and platform build CI                                                                       |
| AMD droplet's latest tested source | Backend `11db2aa999e5d0b88525b7dda23f410aad61f289`; client `579472eb2b6865024c0993eb11c3d94c3901630e`                                                       | Both clean; fresh stable-Diffusers Qwen run and later delivery checks use this pair; earlier GPU runs retain their earlier source identities |
| Local image campaign               | Per-run frozen sources; later publication checkpoint backend `123ede05172fe6049922af014e0bfb5b1dff7aa1` / client `c7af71571d51298427a4ee28205d02f5ccec7efa` | Recorded local ROCm parity, Auto and production-browser results; not reassigned to the latest published pair                                 |
| Windows user's separate checkout   | Not inspected by this Linux review                                                                                                                          | Its source changes and actual GPU results must be retained and reviewed on Windows; available Windows CI is CPU evidence                     |

These identifiers describe application code, not this documentation commit.
The review found no additional uncommitted code on the droplet to merge. The
latest published pair was not redeployed or GPU-tested during the review.

No application tests, installations, model runs or restarts were repeated to
prepare this handoff. Existing source, logs, receipts and CI results were read.
Raw evidence, original failed attempts and source/output hashes remain privately
retained; credentials, host addresses, personal paths, task IDs and generated
media are excluded from this document and Git.

Keep these proof levels distinct:

- **Implemented:** the source supplies the intended behavior.
- **Contract/CPU checked:** unit, schema, genuine tiny-component or model-free
  HTTP tests cover their stated boundaries.
- **Browser checked:** distinguish mocked backends from an actual running app.
- **GPU executed:** the full recorded recipe completed on the recorded machine.
- **Output accepted:** numerical comparison and visual/task review have their
  own outcomes.
- **Platform/resource qualified:** the actual installation, workload, memory
  policy, repetitions and recovery have sufficient evidence on that platform.

A passing task, an attractive historical example and a green CPU suite answer
different questions. None alone qualifies every template or memory setting.

## Approach to preserve throughout the work

MoDiff keeps one effective graph and the existing backend executor. Nodes stay
task- or modality-generic; backend execution specifications select reviewed
model/library adapters, parameter meanings and output normalization. The client
renders those declarations rather than choosing Python implementations.

The simpler image workflow uses real loading, **Encode Inputs**, consumed
**Guidance**, denoising, decoding and preview stages where supported. Optional
**Model Setup**, **Prepare Mask**, **Image Output** and component bundles reuse
ordinary nodes and Block V2 behavior. Grouping reduces visible controls/wires
without creating another pipeline, graph format or runtime cache.

Model/task changes preserve compatible prompts, seeds, inputs, controls and
layout. Unsupported tasks or incompatible guidance/conditioning need a visible
review rather than guessed conversion. An image socket does not make img2img,
instruction editing, masks, ControlNet and ordered references interchangeable.
Opening a saved workflow preserves its authored graph.

The review found this architecture intact. Some curated historical-template
compatibility policy remains in the client template builder; moving its remaining
policy into backend-declared template bindings would improve maintainability.
Retain the existing parity behavior while making that change. See the
[Mellon comparison](simpler-image-workflows-mellon-comparison.md),
[guidance transfer](guidance-control-transfer.md) and
[component bundles](component-bundle-authoring.md).

## Each original feedback point

### 1. The advertised no-script path still calls scripts

**Feedback:** invoking the installer through a Python module did not provide the
ordinary uv/npm setup the reviewer expected.

**Plan and approach:** declare dependencies and accelerator extras in the project,
use real `uv sync` or explicit `uv pip install`, and launch the normal app entry
point. Keep the guided installer available as a separate setup choice.

**Implemented:** [developer setup](developer-setup.md) documents direct commands
that do not dispatch to repository installer wrappers. For a fresh CPU checkout:

```text
uv sync --extra cpu
uv run --extra cpu python -m modiff.preflight --json --check-port 8088 --fail-on-error
uv run --extra cpu python main.py
```

NVIDIA uses `--extra cuda` in each command; Intel uses `--extra xpu`; Apple Silicon
omits the extra. Client development uses `npm ci` and `npm run dev`. The reviewed
Instinct SDK uses `uv venv`, explicit profile/index `uv pip install`, `uv pip check`
and the installed Python directly. A generic sync must not replace vendor wheels.

**Expected acceptance:** a fresh checkout installs and starts from those commands;
preflight recognizes the selected runtime; no installer wrapper or foundational
optional activation is needed. Package setup does not download model weights.

**Recorded result:** native CPU setup/preflight/smokes passed in Windows, Linux
and macOS CI. The AMD droplet used ordinary direct uv/pip and npm setup.

**Remaining:** README clone commands still select GitHub's older default `main`
branches. Until release, explicitly select `fix-ui-ux-issues` in both repositories
and correct setup links. Current-branch installation success does not fix a
fresh-clone instruction that obtains different code.

### 2. `--no-sync` and `--no-project` are redundant

**Plan and approach:** use consistent native project commands and the same
accelerator extra for sync/run. Direct pip-managed environments launch their
installed Python, preserving intentional additions.

**Implemented:** ordinary setup/run instructions need neither flag. Project
isolation can still be meaningful inside managed-tool bootstrap or explicit
verification helpers; those are separate contexts.

**Expected acceptance:** the documented commands preserve the selected runtime
and start successfully without redundant flags or unexpected package replacement.

**Recorded result:** native CPU CI passed the simplified commands; explicit AMD
SDK setup and Python launch passed on the droplet.

**Remaining:** keep platform examples consistent, including Windows paths. Do not
use generic CPU/CUDA sync or auto-syncing `uv run` over the specialized ROCm SDK.

### 3. Windows expandable segments and missing Triton need separate solutions

**Plan and approach:** centralize allocator policy before Torch imports. Establish
ordinary eager/native-SDPA generation independently of optional compilation.
Check a real compiled kernel when an exact workflow requires it.

**Implemented:** [runtime_environment.py](../modiff/runtime_environment.py) uses
Torch's default allocator on Windows, macOS and Linux ROCm. Linux CUDA retains
expandable segments. Explicit operator variables are preserved. Both launch
entry points apply the shared policy before importing Torch.

[runtime_compilation.py](../modiff/runtime_compilation.py) performs a bounded
compiled-kernel probe, including FlexAttention when required. Finding
`torch.compile` or importing Triton is insufficient. Requiring workflows check
the compiler before repository resolution and generation-weight allocation.
Basic image generation remains eager/native SDPA.

**Expected acceptance:** supported Windows image recipes generate with no
injected unsupported allocator option and no Triton activation prerequisite.
Optional compilation either executes a verified kernel or explains its failed
prerequisite while ordinary eager recipes remain usable.

**Recorded result:** source/CPU tests cover policy and compiler boundaries. On
MI300X, expandable segments failed around 19 GiB while much GPU memory remained
free, with about 981 file descriptors under a soft limit of 1,024. Default
allocation reached 32 GiB with eight descriptors; an isolated raised-limit
expandable control also reached 32 GiB. The application fix selects the default
ROCm allocator; it does not change host limits or drivers.

**Remaining:** inspect the separate Windows results for actual eager GPU
generation and allocator behavior. The cloud SDK's Triton installation does not
prove a Triton-absent Windows case. Compile-enabled Windows workflows need their
own compatible toolchain and real-device probe.

### 4. Transformers should be part of normal installation

**Plan and approach:** make Transformers a mandatory base dependency for normal
prompt-conditioned image workflows; keep heavyweight imports lazy.

**Implemented:** `transformers>=5.18.0` is unconditional in
[pyproject.toml](../pyproject.toml). Base-runtime readiness verifies compatibility.
Registry discovery, template browsing and Auto inspection do not install it.

**Expected acceptance:** normal setup prepares tokenizers/text encoders without
a Transformers Install/Activate detour. A damaged environment exposes repair.

**Recorded result:** genuine tiny Diffusers/Transformers interoperability passed
in CPU smokes; full recorded Qwen/Z-Image cloud generation used the installed
base library.

**Remaining:** fix the fresh-clone branch references and review Windows's fresh
installation result. Do not reinterpret successful runs as compatibility with
every future Transformers release.

### 5. PEFT should be installed for ordinary LoRA use

**Plan and approach:** install PEFT with the base runtime, use upstream adapter
APIs and retain exact adapter bytes, names, scales, order and replacement policy.

**Implemented:** `peft>=0.21.2` is unconditional. Native LoRA descriptors support
explicit names and ordered previous adapters feeding the real model owner.

**Expected acceptance:** an ordinary LoRA workflow loads its declared adapters,
changes output when enabled/scaled, supports replacement and restores the
declared set without optional core activation.

**Recorded result:** tiny genuine CPU tests exercised loaded adapter weights,
scale effects and replacement. Historical local full-model LoRA recipe comparisons
retain their source/platform scope. The AMD named-LoRA foundation smoke passed;
downloading a full LoRA artifact alone is not GPU effect/replacement proof.

**Remaining:** inspect or complete actual full-weight LoRA use and replacement on
the admitted Windows/cloud runtime. Keep a compatible minimum plus tested lock
resolution; do not assume an unconstrained latest release works with every stack.

### 6. Remove an unnecessary uv pin and document upgrades

**Plan and approach:** accept operator-installed uv without an exact executable
version requirement. Separate tool updates, ordinary Python requirements, tested
lock resolutions and accelerator ABI constraints.

**Implemented:** the exact uv tool pin is removed. Standalone uv uses
`uv self update`; package-manager installations use their manager. Ordinary
Diffusers now declares `diffusers>=0.41.0`, with the tested stable resolution in
`uv.lock`. Transformers/PEFT also use minimum requirements. Curated model
revisions and reviewed native wheel/ABI constraints remain independently bounded.

**Expected acceptance:** supported operator uv can install the app; upgrades have
a documented path and preserve the chosen accelerator. A library update is
validated before claiming broader compatibility.

**Recorded result:** ordinary native uv setup passed across three CPU CI platforms
and direct vendor-wheel uv setup passed on AMD.

**Remaining:** native lock, managed profiles and direct AMD SDK installations are
different environments. The inspected lock resolves CPU Torch 2.14.1, CUDA
Torch 2.11.0 and Diffusers 0.41.0; managed CPU/CUDA/macOS profiles still select
Torch 2.8; the droplet used Torch 2.10.0 with its ROCm SDK. Their proof must remain
separate. Validate upgrade instructions on a disposable selected-platform setup.

### 7. Refresh/startup spams `Error: write ECONNABORTED`

**Plan and approach:** reproduce the disconnect category, clean up closed sockets
and cancelled requests, preserve task identity across reconnect, and retain real
upstream failure diagnostics. The original message had no stack identifying its
emitting process, so its exact origin remains uncertain.

**Implemented:** Vite's proxy handling classifies known abort/reset/pipe errors
only for owned disconnected browser sockets. Backend sends/callbacks are bounded
and failed sockets are pruned. Client connection generations cancel stale timers
and health probes. Supervisor controls resolve through the configured tunnel.

**Expected acceptance:** repeated refresh reconnects without duplicate tasks or
lost progress/output; expected disconnects do not flood logs; actual backend
unavailability remains visible with recovery guidance.

**Recorded result:** real proxy lifecycle regression coverage is green. Actual
cloud owning-page refresh recovered the same generation, and fresh Qwen execution
and Layered delivery checks recorded zero browser errors. One older deliberate
Stop/replacement/retry journey retained five connection-refused diagnostics before
successful recovery; that journey is not zero-console acceptance.

**Remaining:** inspect Windows production and Vite startup/refresh records
separately. Preserve any full stack for a remaining abort rather than suppressing
all transport errors or assuming every timeout has the same cause.

### 8. Missing Transformers/PEFT immediately produces UI errors

**Plan and approach:** normal setup must install and verify both libraries before
declaring readiness. Existing damaged installations need a coherent base repair
action rather than optional-overlay activation.

**Implemented:** verified base dependencies satisfy their declared runtime
requirements. Missing/incompatible packages are handled by global runtime
readiness: Needs setup, package diagnostics and a native repair command. Auto
blocks unresolved setup before generation. Base-included optional profiles hide
Install/Activate controls.

**Expected acceptance:** fresh setup opens an ordinary image template without
missing foundational runtime warnings. Deliberately damaged private setups show
repair and recover after the documented sync/restart.

**Recorded result:** base/profile/readiness contracts and tiny interoperability
checks passed. Cloud full-model execution used the standard installed libraries.

**Remaining:** inspect fresh/damaged Windows setup and the actual UI result.
Correcting clone references is necessary so users receive this implementation.

### 9. Installation/Activate completion is unclear and slow without status

**Plan and approach:** remove the core-library activation ceremony. For genuine
extra runtimes, keep one durable backend job with explicit installation,
verification, cutover and ready/failed states that survive refresh.

**Implemented:** the client resumes the durable job and shows its progress.
Activation completes only after the replacement worker has the exact expected
environment/profile/spec identity and ready runtime. Foundation libraries do
not require this optional flow.

**Expected acceptance:** the user always sees the current stage and actionable
failure; Activate does not appear while cutover remains active; readiness is
confirmed before generation. Refresh/reopen must resume the same operation.

**Recorded result:** job/state/restart contracts and transient reconnect tests
passed. Direct AMD setup is not an actual optional Install → Activate lifecycle.
A full real optional install/activation/generation journey remains unqualified.

**Remaining defect:** client job polling catches every error, including permanent
404, malformed payload and identity mismatch, then retries with stale progress.
Expose connection/protocol state and safe reconciliation/retry while retaining
the original job identity. A polling failure cannot fabricate a terminal backend
failure or authorize another installation while the outcome is unknown.

### 10. The second generation fails Auto's memory calculation

**Feedback:** `Combined systemRamBytes: needs 8589934592 bytes; 4279840768 bytes
available`. The reported field is **system RAM**, even though the reviewer
described a suspected VRAM calculation issue. The original OS, model, placement
and complete receipt were not available; nominal hardware was reported as
32 GB RAM / 16 GB VRAM.

**Plan and approach:** distinguish machine-capacity eligibility from additional
working memory. Count actual model owners, credit only compatible live storage,
respect physical CPU/GPU/shared pools and reclaim only app-owned references.
Replan against measured free memory after release.

**Implemented:** machine tiers are checked against total capacity, rather than
charged as another allocation after model loading. Exact cache identity and
measured unique weight storage determine reuse credit. CUDA reservations are not
added again to free memory. Projected reclaim can authorize preparation, but
dispatch uses actual post-release capacity. Auxiliaries and adapters are counted.

**Expected acceptance:** cold and genuine warm/recompute runs, changed prompt/
seed, model changes and safe release/retry keep the full creative recipe. Combined
owners fit or receive an actionable plan before allocation; a cached preview
does not establish another model forward.

**Recorded result:** local Auto cold/warm, masked output parity and genuine
resident recomputation passed at recorded sources. Cloud seed-mutated generation,
task/model changes and a two-owner full recipe completed on the large GPU.

**Remaining:** the original free-RAM value is 3.985912 GiB, about 14.43 MiB below
the unchanged 4 GiB host headroom floor. That observation can still legitimately
reject unless measured owned reclamation crosses the floor. Dedicated GPU-only
weights cannot invent host-RAM credit.

A separate source review found a **combined-fit blind spot**: built-in
machine-capacity recipes without `workingMemoryRequirements` contribute zero
primary-owner RAM/VRAM demand, followed by one workflow headroom floor. Distinct
native resident models can therefore be admitted without budgeting their combined
weight/activation storage or selecting release scheduling. The retained executor
rechecks auxiliaries but not each primary native loader. Native ComponentsManager
resident placement does not use the standard `mm_load` eviction path.

This is a confirmed static planning gap and plausible overcommit risk, not a newly
observed OOM. Add separate working demands or withhold combined-fit authority when
unknown. Do not sum machine tiers again or lower the floor to fit the guessed case.

## Image templates and output comparisons

All **54 fresh image templates** use the ordinary operation-authoring transaction.
Their visible graph is the graph submitted to Run, with exact reviewed selections
and preserved creator prompts, inputs, seeds and settings.

| Native-stage family                                | Recipes |
| -------------------------------------------------- | ------: |
| Z-Image Turbo                                      |       7 |
| Qwen Image                                         |      10 |
| Qwen Image Edit, including masked/outpaint recipes |       7 |
| Qwen Image Edit Plus                               |       7 |
| Qwen Image Layered                                 |       1 |
| FLUX Schnell/Dev/Krea                              |      11 |
| FLUX Kontext                                       |       2 |
| FLUX.2 Klein                                       |       3 |
| **Total with real native stages**                  |  **48** |

The other six are two FLUX Fill, two Canny/Depth and two Redux recipes. They
retain explicit whole-pipeline operations because the selected native task
contracts do not supply equivalent masked/control/prior conditioning. Cosmetic
Encode Inputs or Guidance nodes would not implement those missing operations.

Parity work preserves actually consumed settings, not just similar field names:
true CFG versus distilled guidance, enabled/formulation state, scheduler/time
arithmetic, tokenizer masks, generator behavior, dtype, offload, original reference
order, masks/outpaint geometry and named LoRA sets. Qwen masked compatibility is
explicitly selected and owner-bound; ordinary starters retain native defaults.
See [image validation](image-template-validation.md) for those boundaries.

The historical comparison index records **66 comparisons across 52 recipes**:
59 passing records, six retained failures and one historical incomplete record,
with 70 ordered output pairs. Each of those 52 recipes has a scoped passing
comparison. Two additional ControlNet recipes have current-only functional
repairs, because an old successful output was unavailable. The snapshot contains
141 completed tasks and 148 ordered captures.

Later local work completed all four original masked recipes under Custom and
ordinary Auto, resident object recomputation, production Gallery object parity,
and full 50-step Qwen raw/component-bundle/restored comparisons. The five-run T2I
sequence captured 500 predictions and 250 scheduler updates, prompt/seed changes
and same-task active refresh recovery. This work was done and logged; a broader
remaining qualification gate does not erase it.

These results do not prove all 54 recipes on the latest stable runtime, every
setting or Windows. An old saved workflow is preserved, rather than automatically
rewritten into a new template. Historical Gallery examples remain labelled
**Previous recipe example** when their execution selection has changed.

## Existing tests: expected versus recorded results

### Installation, code and browser coverage

| Check                                                                                          | Expected result                                                                                          | Recorded result and limit                                                                                    |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Native CPU setup: `uv sync --extra cpu --locked`, test requirements, `uv pip check`, preflight | Dependency consistency and ready standard runtime                                                        | Passed in backend CI on Windows/Linux/macOS; CPU scope                                                       |
| `scripts/smoke_base_runtime.py`                                                                | Tiny genuine generation plus Transformers/PEFT and named-LoRA interoperability, repeat/scale/replacement | Passed; no production weights or GPU fit proof                                                               |
| `scripts/smoke_service_package.py`                                                             | Existing HTTP/queue/service execution using isolated temporary storage                                   | Passed; model-free scope                                                                                     |
| Full local backend `python -m pytest -q`                                                       | Current CPU/source regression gate green with explicit skips                                             | 5,291 passed, 31 skipped, 11,331 subtests passed at the published code checkpoint                            |
| Client `npm ci`, `npm run check`, `npm run check:ui`                                           | Locked installation, lint/types/contracts/build/browser regressions pass                                 | Existing final quality/browser gates and CI passed; mocked cases do not qualify generation                   |
| Windows/macOS client platform smoke                                                            | Package-lock/license/typecheck/build work on those OSes                                                  | Passed; separate from Windows GPU execution                                                                  |
| Gallery acceptance / `gallery:verify`                                                          | Current example recipe, media, complete provenance and technical gates pass                              | Last retained report failed with 66 errors over 70 examples; standard quality gate passing does not close it |

Completed CI evidence for the reviewed code:

- [Backend run 37847522164](https://github.com/sdevil7th/MoDiff/actions/runs/37847522164):
  lint and Windows/Linux/macOS backend jobs all succeeded at `5f38a40`.
- [Client run 37848002505](https://github.com/sdevil7th/MoDiff-client/actions/runs/37848002505):
  quality and Windows/macOS platform jobs succeeded at `bfad813`, paired with
  backend `5f38a40`.

These runs predate this documentation change. Future changes need their own
appropriate checks; preserving the old receipts does not relabel them as new runs.

### Full-model and workflow evidence

| Machine/workload                                             | Expected result                                                                             | Recorded result and limit                                                                                                                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Local Radeon 8060S/gfx1151 shared-memory ROCm image campaign | Original recipe preserved; matched old/new ordered outputs; actual Auto/recompute behavior  | Scoped parity and repairs described above; shared memory does not qualify dedicated 16 GB Windows VRAM                                                                             |
| Local Qwen Control cold/warm Auto                            | Actual model reuse and fresh required execution without repeated loading                    | Four original full runs; 144 ControlNet forwards, 288 base predictions, 144 scheduler updates; exact decoded repeatability; current-only repairs                                   |
| MI300X Z-Image Turbo                                         | Full 1024-square BF16/no-quant 8-step recipe, preview and refresh                           | Completed after the allocator correction; initial HIP failure retained                                                                                                             |
| MI300X Qwen Image 2512                                       | Full 1024-square BF16/no-quant 50-step T2I, real seed change, preview/refresh               | Completed; this large-memory result does not reproduce the original low-free-RAM host                                                                                              |
| Qwen T2I → img2img by Load Image/task adaptation             | Immediate readiness; same external image feeds required stages; authored strength consumed  | Ordinary upload without refresh passed; authored 50 steps/strength 0.5 yielded 25 effective updates                                                                                |
| Qwen img2img → Edit 2511 by model/task Apply                 | Preserve compatible edits/input; correct new model and required image wires; full execution | Full 40-step execution and UI recovery passed; inherited complex prompt output failed visual review                                                                                |
| Independent Qwen producer → Edit consumer                    | Both actual owners and sinks, explicit typed media chain, full 50+40 steps                  | Completed; peak Torch allocation about 111.83 GiB; consumer quality failed and smaller-GPU combined fit remains unproved                                                           |
| Qwen Edit 2511 canonical single-reference                    | Requested full edit with original source/recipe                                             | Full 40-step execution and separate visual inspection recorded; not interchangeable with the failed two-reference recipe                                                           |
| Fresh stable 0.41 Qwen two-reference edit                    | Original prompts/ordered inputs, 40 steps, seed 5103, true CFG 4; usable output and refresh | One ordinary Auto Run completed with zero recorded browser errors; peak allocation 62,294,398,976 bytes; strong speckles and absent requested lavender sprig mean visual FAIL      |
| Same complex Qwen whole/native-math/vision/MM-token controls | Distinguish integration, attention and processor hypotheses                                 | Full execution completed but artifacts/instruction failure remained; no demonstrated native-only defect or hardware cause                                                          |
| Qwen Layered                                                 | Full 30-step, three ordered 640-square RGBA outputs with alpha/media identity               | Original and separately labelled substituted-source runs completed; later full-size collection navigation/refresh passed; count/composite alone is not semantic quality acceptance |
| Active refresh and Stop/retry                                | Same task survives refresh; Stop reaches terminal cancellation before fresh retry           | Scoped real passes; one worker-replacement journey retained connection diagnostics; not universal zero-error recovery                                                              |
| Manual wiring, Undo/Redo, Save/reopen                        | Same authored graph/settings persist and later Run executes it                              | Contract/mocked coverage and actual prepare-only journeys exist; not every saved/reopened modification received a full GPU Run                                                     |

### Large checkpoints and access boundaries

| Model                    | Expected checkpoint                                                                 | Actual outcome                                                                                                                                                                                                                                                  |
| ------------------------ | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLaDA2.1 Flash           | Load original 205.8 GB BF16 weights and exercise native Diffusers                   | Standalone **text** smoke completed one 32-token block/32 forwards, 28 GPU decoder layers plus four CPU-offloaded layers; peak Torch GPU allocation 179.77 GB and process RSS 199.95 GB; answer truncated before its result; no MoDiff/image/Auto qualification |
| Cosmos3-Super-Text2Image | Full 131.3 GB active image selection, mandatory guardrails, original 50-step recipe | Generation weights downloaded/verified; legitimate gated guardrail access pending; no completed generation                                                                                                                                                      |
| FLUX.2-dev               | Exact admitted full image route and complete model access                           | Required gated access pending; no completed generation                                                                                                                                                                                                          |
| MiniMax H3               | One selected full 144 GB audio/video workflow                                       | Not downloaded/executed; publisher territorial eligibility must be resolved for the compute environment                                                                                                                                                         |

The bounded metadata survey did not identify a reviewed single native Diffusers
image checkpoint with 200 GB–1 TB of required weights. Repository totals can
include duplicate precision formats and task variants. LLaDA meets the large
weight-size target but is text, not image. See
[large-model validation](large-image-model-validation.md) and
[AMD cloud validation](amd-cloud-qualification.md).

## Problems found and fixes already made

| Diagnosed issue                                                                                 | Implemented correction                                                              | Evidence boundary                                                                     |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Direct Instinct SDK installation misidentified as another AMD profile                           | Detect the reviewed dedicated gfx942 runtime without requiring an installer receipt | Source/profile tests and direct cloud readiness; profile remains preview              |
| ROCm expandable-segment HIP allocation failed despite free VRAM                                 | Shared pre-Torch default allocator policy for ROCm                                  | Controlled allocation/FD observations and subsequent full Z-Image execution           |
| Frontend used the wrong supervisor control endpoint through the cloud tunnel                    | Resolve control address through configured backend transport                        | Source/browser checks and real scoped recovery                                        |
| Uploaded image left workflow readiness stale; task/model changes lost media roles               | Reactive readiness and generic semantic-role migration/fanout preservation          | Regression and real upload/I2I/model-change execution                                 |
| Legacy processor selected wrong config; Qwen encoder omitted supplied multimodal token identity | Selected config binding and signature-aware backend processor bridge                | Genuine tiny model tests and full run; bridge did not cure the complex visual failure |
| Layered Gallery full-size preview lost ordered collection navigation                            | Preserve complete ordered media collection and current index                        | Real retained-output delivery/navigation/refresh; no new generation implied           |
| Windows extended Hub cache paths failed validation                                              | Normalize and validate the actual Windows extended-path representation              | Focused tests and later green Windows CPU CI                                          |
| Reviewed whole-workflow loader had blank workflow binding; generated route identities drifted   | Bind exact already reviewed selection and regenerate coordinated strict identities  | Source/catalog/regression checks; no new unrestricted admission or GPU claim          |

Capture-selector mistakes, private runner gates, serialization errors, archive
timeouts/hardlinks, copied read-only fixture failures and missing test browsers
were separately retained as tooling/fixture issues. Their correction does not
turn an unrelated application or visual failure into a pass.

## Remaining issues and brief fix plans

### P1: fresh-clone instructions obtain older code

Correct both READMEs, stale client setup links and the Windows fashion guide to
select the paired reviewed branch until it is released. Keep the branch explicit
in fresh-checkout examples. Review commands for the same accelerator extra and
correct Windows executable paths. Acceptance: fresh native instructions obtain
the intended code and install/start without wrappers or core activation.

Relevant source: [backend README](../README.md), client README/setup links, and
[Windows fashion guide](windows-fashion-demo.md). Default branches were still
older `main` heads when reviewed; no release merge was performed by this handoff.

### P1: incomplete combined Auto working-memory authority

Extend existing recipe/owner planning with separate working-storage, load-transient
and activation requirements for the actual placement/workload. Deduplicate proven
shared storage; count independent owners. Make uncertain combined fit explicit
instead of interpreting one headroom floor as their storage budget. Use the
existing executor's safe last-use releases and actual pre-allocation resampling.

Add focused coverage for distinct native resident owners, shared components,
mixed placements, unknown demand and primary-loader pre-allocation checks. Later
hardware acceptance must exercise an unchanged full recipe on a constrained
device. Preserve creative settings and the corrected capacity/free distinction.
Review mixed ComponentsManager offload ownership separately before promising
unrestricted combinations.

Relevant source: [workflow planner](../modiff/workflow_auto_resource.py),
[owner lifecycle](../modiff/workflow_auto_lifecycle.py),
[resource recipes](../modiff/auto_resource.py) and
[native loaders](../modules/ModularDiffusers/loaders.py).

### P1: current Gallery acceptance has 66 errors

| Error class                           | Count | Required repair                                                                                                                                                                                 |
| ------------------------------------- | ----: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stale template lock hashes            |    55 | Derive current proof from matching consumed recipes and complete existing receipts; execute only missing affected cases later                                                                   |
| Stale prompt hashes                   |     2 | Bind evidence to actual current consumed prompts; retain historical examples where they differ                                                                                                  |
| Audio duration mismatches             |     2 | Validate actual route/frame-quantized duration and template intent; retain the observed 15/30-second outputs against 90/75-second expectations; do not pad or widen tolerances to claim success |
| Missing Ghibli LoRA example media     |     1 | Recover and verify the approved asset/provenance or obtain a genuinely accepted replacement through the normal asset flow                                                                       |
| Low-VRAM Qwen model/provenance errors |     6 | Reconstruct the complete actually executed model set from independent receipts and validate duplicate comparison claims                                                                         |

The report covered 70 examples, including non-image media. Stale hashes are
missing current-recipe proof, not evidence that all 54 image graphs fail. Keep
historical provenance visible; merely changing hashes cannot qualify new graphs.
Acceptance: the current Gallery gate passes with complete media/task/model/recipe
evidence and independent quality review.

### P2: optional-runtime polling hides persistent failures

In the client's `RuntimeOptimizationsCard`, distinguish abort/unmount, transient
connection loss and permanent/protocol failure. Display polling state, reconcile
the durable job/catalog through a safe Retry, and retain identity throughout.
Do not infer job termination from a connection error. Cover 404, malformed response,
identity mismatch, recovery after restart and prevention of duplicate mutations.
Acceptance: no indefinite unexplained stale busy state, and real backend readiness
still controls activation completion.

### P2: complex Qwen output quality remains unacceptable

Keep the failed original output and controls. No reviewed source comparison
established a native-only error; whole execution also fails this recipe. A later
single-variable publisher-style conditioning control may help distinguish recipe
effects, but it must preserve the original positive prompt, inputs, seed, steps,
CFG and precision. Record differences as another recipe, not an improvement to
the original until visually established. Avoid speculative default changes or
claims that the processor bridge fixes quality.

### P3: test-browser prerequisite ordering

Move Chromium installation before `npm run check` in client Windows guidance and
make the frontend contributor prerequisites clear in setup links. `npm ci` itself
passed on AMD; its initial quality failure was missing Chromium. Acceptance:
documented fresh contributor checks reach the gate with browser prerequisites
available, matching existing CI/CONTRIBUTING ordering.

### Remaining qualification work

- Review actual Windows installation/GPU receipts after source integration; CI
  cannot replace that review.
- Complete route-specific cold/warm/recompute, prompt change, model-return,
  saved/reopened Run, full-weight LoRA replacement and resource-limited recovery
  only where existing evidence is incomplete.
- Qualify real optional Install → verify → Activate → restart → generation on
  an admitted extra runtime; base Transformers/PEFT need none of those steps.
- Resolve legitimate gated access and runtime closure before Cosmos/FLUX.2 work.
  Respect model eligibility and mandatory guardrails.
- Refresh exact-template/current-runtime evidence only for affected routes;
  preserve prior completed comparisons and failed attempts.

The first implementation batch can handle setup references, browser ordering and
optional-job diagnostics without expensive model runs. The next batch should
address combined Auto demand with focused planner/executor checks. Gallery evidence
reconciliation can reuse trustworthy retained results; missing output/quality/
hardware checks are a separate bounded campaign after the Windows merge.

## Windows comparison and merge procedure

1. In **each repository**, preserve current Windows source, branch/commit,
   uncommitted changes and test evidence. Create a separate integration branch
   and commit only reviewed source files there. Keep secrets, config, model
   caches, virtualenvs, outputs and raw logs outside Git. If preserving a dirty
   tree separately, retain a recoverable before-copy before proceeding.
2. Fetch origin and compare the Windows branch with the reviewed branch:

   ```powershell
   git fetch origin
   git log --oneline --left-right HEAD...origin/fix-ui-ux-issues
   git diff --stat HEAD origin/fix-ui-ux-issues
   ```

3. With a clean source checkpoint, merge into the Windows integration branch
   while retaining an opportunity to inspect the merge before committing:

   ```powershell
   git merge --no-ff --no-commit origin/fix-ui-ux-issues
   git status --short
   git diff --cached --stat
   git diff --cached --check
   ```

   Resolve semantic conflicts across backend/client together. Inspect both sides
   of dependency, Auto, model-adapter, template and guidance changes. Avoid
   blanket ours/theirs resolution. An already integrated branch may report
   up to date, in which case there is no new merge to commit.

4. Derive generated catalogs and assets from the combined authored source when
   affected. Build the client through its normal tooling; mirror its output into
   backend `web/` while preserving `web/user/`. Do not merge minified bundles or
   pick arbitrary old identity hashes. Refresh dependent inventories through
   their existing generators and confirm no qualification/status promotion.
5. Recreate/repair the runtime on Windows with its actual accelerator profile
   before the applicable contributor checks. Never copy Linux environments or
   update an environment while its worker is generating. Preserve candidate source
   and build identities for those checks; follow the existing setup/validation
   guides for dependencies and isolation.
6. After reviewing conflicts, generated changes and the applicable contributor
   checks, finish the merges explicitly. Commit the reviewed backend merge first.
   If publishing a new paired candidate, update the client's immutable CI backend
   reference to that actual backend commit, then commit the reviewed client merge.
   Record both resulting SHAs. When Git reported up to date, retain the existing
   commit instead of inventing a merge commit. Check both source trees are clean
   and verify the served frontend matches the recorded paired source/build before
   live validation.
7. Use [Windows acceptance](windows-app-validation.md) and
   [three-machine validation](three-machine-app-validation.md) for the subsequent
   checks. Start with eager generation, native setup, repeated actual runs,
   refresh, input attachment, model/task edits, LoRA and Auto memory observations.
   Keep compilation and access-blocked large routes separate.

Windows tests made before the merge remain evidence for their original source.
A successful Git merge does not establish generation or resource qualification.
This document adds no executable contract, qualification flag, new model
admission or runtime change.
