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

| Source                           | Reviewed checkpoint                                                                                                                                         | Evidence boundary                                                                                                                                                                                          |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Published full-image backend     | `3b7cab86f5ddb17baf7a34d5b98ff40efb2736f1`, branch `fix-ui-ux-issues` | Two exact full-weight image recipes, 202 canonical graphs, regenerated source metadata and mandatory partial-stage Cosmos safety; final local CPU gate: 5,382 passed, 43 skipped, 11,399 subtests |
| Published full-image client      | `6d54bf8028d7a717d883bceb1f124a1f1abf8835`, same branch | 80 public/56 image recipes, exact variant labels and clearly marked editorial cards; complete quality gate, full prior browser regression and final focused card test passed; immutable CI backend pairing is `3b7cab8` |
| Reviewed backend core            | `18445cf985732c3c59d22b5931d72150fa49a284`, branch `fix-ui-ux-issues`                                                                                       | Combined-owner Auto guards, zero-free telemetry, original Cosmos caption and authoritative task dependencies; local full CPU suite passed; then-current Windows pytest exposed the verifier defect subsequently corrected at `4432062` below |
| Reviewed client recovery         | `3e4f49c627f992ab7ec5b6ec3330d849255c2b2e`, same branch                                                                                                     | Bounded optional-job recovery and empty-UI-state authoring correction; full check, 3 shared-control and 254 mocked Studio cases passed; immutable CI backend pairing points to `18445cf`                   |
| AMD Cosmos runtime lifecycle     | Backend `18445cf985732c3c59d22b5931d72150fa49a284`; client `3e4f49c627f992ab7ec5b6ec3330d849255c2b2e`                                                       | Real Setup installation, same-job refresh, staged validation, activation and replacement-worker readiness passed; model inference remains a separate boundary                                              |
| AMD full FLUX.2 execution        | Backend `ad79f3a8067febb338980e2838cec1bc7086b199`; client `bfad8134b82f6b77affbd10599a71740c578cae1`                                                       | Frozen clean pair used for the full original-weight FLUX.2 runs below; later source changes do not inherit those receipts                                                                                  |
| AMD original Cosmos template     | Backend `3b7cab86f5ddb17baf7a34d5b98ff40efb2736f1`; client `6d54bf8028d7a717d883bceb1f124a1f1abf8835` | Ordinary Template Create, original cold and changed-seed Custom recipes, mandatory Decode/AfterDecode, fresh media and bounded visual review passed; Auto and Gallery remain unqualified |
| AMD original FLUX template       | Backend `3b7cab86f5ddb17baf7a34d5b98ff40efb2736f1`; client `6d54bf8028d7a717d883bceb1f124a1f1abf8835` | Ordinary Template Create and original configured 50-step Custom cold Run passed; original PNG byte-identical to the accepted historical native output; editing, Auto and Gallery remain separate |
| AMD Qwen execution and delivery  | Backend `11db2aa999e5d0b88525b7dda23f410aad61f289`; client `579472eb2b6865024c0993eb11c3d94c3901630e`                                                       | Fresh stable-Diffusers Qwen run, negative-conditioning control and retained Layered delivery use this pair; earlier GPU runs retain their earlier identities                                               |
| Local image campaign             | Per-run frozen sources; later publication checkpoint backend `123ede05172fe6049922af014e0bfb5b1dff7aa1` / client `c7af71571d51298427a4ee28205d02f5ccec7efa` | Recorded local ROCm parity, Auto and production-browser results; not reassigned to the latest published pair                                                                                               |
| Windows user's separate checkout | Separate GPU checkout not inspected; available Windows CPU CI reviewed                                                                                                                          | Its source changes and actual GPU results must be retained and reviewed on Windows; available Windows CI is CPU evidence                                                                                   |

These identifiers describe application code, not this documentation commit.
The initial review found no additional uncommitted droplet code to merge. Its
historical results were inspected without repeating model runs. A subsequent
six-hour image completion block added the focused fixes and new executions
explicitly described below. The frozen source pair on each receipt remains the
authority; a later publication does not relabel earlier execution evidence.

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

**Follow-up implemented:** both READMEs and the Windows fashion guide now clone
`fix-ui-ux-issues` explicitly, and client setup links select that branch. Existing
command-contract assertions check the paired commands. This corrects instructions
that previously obtained GitHub's older default `main` code; no release merge
into the default branches is implied.

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

**Remaining:** review Windows's actual fresh GPU installation result; the clone
references are now corrected. Do not reinterpret successful runs as compatibility
with every future Transformers release.

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

**Diagnostic follow-up:** remaining Qwen quantization and LTX condition-import
errors now direct missing or incompatible foundational packages to the matching
accelerator's base repair instructions. Text, vision and speech profile descriptions
also call Transformers a standard installation. Base-overlay rejection preserves
the accelerator selection rather than recommending a bare sync that could replace
specialized vendor wheels. Genuine additional runtimes retain their own actions.

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
passed. On MI300X, the ordinary Setup card installed Cosmos Guardrail 0.3.1 in
17.1 seconds. One refresh during installation resumed the same durable job;
isolated validation passed and the staged Activate action appeared. A single
ordinary Activate completed in 36.1 seconds, including visible restart/retry
progress and ready status from the replacement worker with the exact active
environment/spec. There was one Install and one Activate, with an idle queue
after completion. This closes that package/UI lifecycle on the recorded Linux
runtime; safety-model inference, generation and Windows require separate proof.

**Follow-up implemented:** permanent 404, malformed payload, identity mismatch
and older status timestamps pause with a visible error and last known progress.
Transient failures, including a non-JSON HTTP 503 during restart, use bounded
backoff. **Retry status** reads the same durable job; it does not repeat
Install/Activate. Actual rendered mocked cases cover those failures, recovery
and duplicate-mutation prevention. The real installation/activation receipts
are preserved as two connected journeys: the installation recorder stopped
because its private assertion misunderstood the cheap catalog's intentional
`staged_unchecked` status. A separately bound activation continuation completed
without reinstalling or bypassing integrity validation. Inference remains
separately gated.

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

**Follow-up implemented:** unknown independent primary-owner demand no longer
advertises combined fit from capacity tiers and one headroom floor. Proven safe
sequential lifetimes use existing owner release and a real pre-allocation memory
resample. Unknown overlapping, opaque/custom or non-releasable lifetimes reject
Auto with actionable guidance. Single-owner warm reuse and complete explicit
working budgets retain their existing behavior. Genuine executor CPU tests verify
that the prior owner is destroyed before the next loader allocates, while its
decoded pixels survive. This closes the diagnosed planning authority gap; real
constrained-device release remains a separate hardware acceptance check.

A further diagnosed false blocker charged the minimum tier's SSD-offload disk
floor even when the accepted full-residency tier selected `none`. The planner now
uses the accepted residency tier's disk requirement unless an explicit placement
requirement applies. Actual group-disk offload and missing-artifact/download
capacity checks remain enforced.

Known zero free device memory now also retains priority over a fallback
process-local estimate. A genuine XPU hardware-probe/next-owner regression
demonstrated that the old truthiness fallback could admit an allocation into an
exhausted device pool. CUDA/MPS partial snapshots receive the same correction;
unknown measurements and existing shared-pool rules retain their prior fallback.
This is a reproduced CPU contract defect, not an observed MI300X OOM.

**Local ownership follow-up:** native global model-CPU hooks and resident owners
now require the existing release schedule when their lifetimes are separate;
incompatible overlap is rejected before loading. Genuine group/global hooks on
the same device retain their supported connected-loader path. Explicit borrowed
components are checked before allocation and keep their source placement,
metadata and disk-file ownership, including resident ControlNet under group
offload. A separate genuine shared-storage regression fixes reuse credit being
consumed by an unrelated cached owner: compatible current owners are selected
before physical bytes are deduplicated. Pre-run CUDA cleanup also preserves
known zero free memory. Focused checks passed 530 tests and 510 subtests on a
consistent CPU runtime; these fixes still need live GPU ownership acceptance.

## Image templates and output comparisons

The historical **54-template image migration** uses the ordinary
operation-authoring transaction. Its visible graph is the graph submitted to Run,
with exact reviewed selections and preserved creator prompts, inputs, seeds and
settings. The current source catalog has **80 public templates, including 56
image templates**: the original 54 plus full FLUX.2-dev and original Cosmos3 Super
native-stage recipes. Those two additions are advanced, unverified Custom
recipes with editorial illustrations, without generated Gallery examples or
Auto qualification. The family counts below describe the unchanged historical
54-recipe comparison set.

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
setting or Windows. The separate full FLUX.2-dev and Cosmos3 Super public source
recipes use advanced, unverified Custom selections with exact revisions and
mandatory prerequisites. Their source-contract admission does not qualify
ordinary template creation, generation, Auto or Gallery media. Both new
templates completed ordinary Create and original configured 50-step Runs on
the final pair: Cosmos cold/changed-seed and FLUX cold, with accepted outputs
recorded below. This does not promote their catalog status or Auto/Gallery. Keep the original 54-recipe comparison set intact and record
each new template's actual Create and Run separately. An old saved workflow is preserved, rather than automatically
rewritten into a new template. Historical Gallery examples remain labelled
**Previous recipe example** when their execution selection has changed.

## Existing tests: expected versus recorded results

The final full-image publication above passed the native CPU suite in 490.30
seconds. Its preceding attempt is retained: four checks still expected the old
200/78/52 source inventory, and one treated the required SDK side-output terminal
as unused. The corrected tests explicitly preserve hidden 148, Gallery 70 and
the historical 54 image recipes, enforce the new recipes' unverified Custom
boundaries, and recognize a terminal only through the backend's declared roles
and complete matching graph. Missing, duplicate, foreign-owner or undeclared
edges/nodes still fail; no Cosmos stage was deleted to satisfy the check.

The published client's complete `npm run check` passed. The browser regression
passed three shared-control and 255 mocked Studio cases before the final artwork
addition; the final mounted variant/card test then passed with real local PNG
decoding. The actual emitted bundle has 108 files (4,862,356 bytes), with index
SHA-256 `43aae210b49e9a5fdd5d7d1b187129fbb21bdfc5116070960448ad2eb0c15aae`.
The mirrored backend bundle matches every emitted byte and preserves user files.
The model-free HTTP/service smoke passed with zero model downloads. These checks
remain distinct from actual template GPU generation and constrained-memory Auto.

### Installation, code and browser coverage

| Check                                                                                          | Expected result                                                                                          | Recorded result and limit                                                                                                                                                                                            |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Native CPU setup: `uv sync --extra cpu --locked`, test requirements, `uv pip check`, preflight | Dependency consistency and ready standard runtime                                                        | Passed in backend CI on Windows/Linux/macOS; CPU scope                                                                                                                                                               |
| `scripts/smoke_base_runtime.py`                                                                | Tiny genuine generation plus Transformers/PEFT and named-LoRA interoperability, repeat/scale/replacement | Passed; no production weights or GPU fit proof                                                                                                                                                                       |
| `scripts/smoke_service_package.py`                                                             | Existing HTTP/queue/service execution using isolated temporary storage                                   | Passed; model-free scope                                                                                                                                                                                             |
| Historical full local backend `python -m pytest -q`                                            | CPU/source regression gate green with explicit skips                                                     | 5,291 passed, 31 skipped, 11,331 subtests passed at `5f38a40`; retained historical evidence                                                                                                                             |
| Final full-image backend `python -m pytest -q`                                                 | Current source and exact catalog regressions green with explicit skips                                   | 5,382 passed, 43 skipped, 11,399 subtests passed in 490.30 seconds at `3b7cab8`; native CPU scope                                                                                                                        |
| Follow-up native CPU backend gate                                                              | Current Auto/setup/runtime regressions with genuine installed stable Diffusers and exact catalog fixture | Verifier-fix checkpoint `cc6f54e`: 5,327 passed, 43 skipped and 11,392 subtests passed in 586.34 seconds. Earlier CLI import-path failures and stale bundle-ledger failure remain retained; their diagnosed corrections preceded this full green run |
| Historical client `npm ci`, `npm run check`, `npm run check:ui`                                | Locked installation, lint/types/contracts/build/browser regressions pass                                 | Published `bfad813` quality/browser gates and CI passed; current local publication gates are recorded above; mocked cases do not qualify generation                                                                      |
| Follow-up optional-runtime client gate                                                         | Visible safe status recovery, no duplicate mutation, complete UI regression coverage                     | Full `npm run check`, HTTP service-package smoke and UI: 3 shared-control plus 254 mocked Studio tests passed. Real optional install/refresh/activation also passed on the frozen pair; generation remains separate  |
| Windows/macOS client platform smoke                                                            | Package-lock/license/typecheck/build work on those OSes                                                  | Passed; separate from Windows GPU execution                                                                                                                                                                          |
| Gallery acceptance / `gallery:verify`                                                          | Current example recipe, media, complete provenance and technical gates pass                              | Last retained report failed with 66 errors over 70 examples; standard quality gate passing does not close it                                                                                                         |

Completed CI evidence for the reviewed code:

- [Backend run 37847522164](https://github.com/sdevil7th/MoDiff/actions/runs/37847522164):
  lint and Windows/Linux/macOS backend jobs all succeeded at `5f38a40`.
- [Client run 37848002505](https://github.com/sdevil7th/MoDiff-client/actions/runs/37848002505):
  quality and Windows/macOS platform jobs succeeded at `bfad813`, paired with
  backend `5f38a40`.
- [Follow-up backend run 37894213823](https://github.com/sdevil7th/MoDiff/actions/runs/37894213823):
  lint and Windows/Linux/macOS native CPU setup, preflight, tiny base/LoRA smoke,
  HTTP service smoke and full pytest jobs succeeded at `ad79f3a`.
- Final full-image backend [push run 37922619939](https://github.com/sdevil7th/MoDiff/actions/runs/37922619939)
  and [PR run 37922624852](https://github.com/sdevil7th/MoDiff/actions/runs/37922624852)
  both succeeded at `3b7cab8`, including ordinary uv setup, dependency checks,
  preflight, runtime/service smokes and required lint. Each Windows suite passed
  5,393 tests and 11,398 subtests with 32 skips; each Linux/macOS suite passed
  5,379 tests and 11,399 subtests with 46 skips. These platform totals remain
  separate from the local native CPU environment's 5,382 passes and 43 skips.

- Final client [push run 37922676741](https://github.com/sdevil7th/MoDiff-client/actions/runs/37922676741)
  and [PR run 37922683061](https://github.com/sdevil7th/MoDiff-client/actions/runs/37922683061)
  use `6d54bf8` and backend `3b7cab8`. Both passed `npm run check` and
  Windows/macOS platform smokes; their full browser CI suites were still running
  at the six-hour cutoff, **12:01:52 UTC**. Both later exceeded their configured
  one-hour job limit while still progressing. The shared-control tests completed;
  the mocked suites did not produce a final result. Local browser results above
  remain independently completed evidence; these cancelled CI runs are not passes.
  The browser gate now has a separate job with its own native backend setup and
  production build. The complete quality gate, browser tests, retry settings and
  one-hour limits remain intact; failure or cancellation requests evidence upload.
  The revised CI needs its own completed run after publication.

These runs predate this documentation change. Future changes need their own
appropriate checks; preserving the old receipts does not relabel them as new runs.

### Local completion checks on 9 October 2026

The follow-up changes were reviewed against backend `8eba6fb` and client
`6d54bf8`, without repeating production model runs. A separate clean CPU copy
passed native `uv sync --extra cpu --locked`, explicit test-dependency
installation, `uv pip check` and ordinary preflight. The consistent runtime was
Torch 2.14.1 CPU, Diffusers 0.41.0, Transformers 5.18.0 and PEFT 0.21.2.
The real tiny eager generation/LoRA smoke passed repeat generation, named adapter
weights, scale changes and replacement. The isolated HTTP/service smoke passed
with zero model downloads. A separate clean `npm ci` on Node 24.12.0/npm 11.6.2
installed the reviewed lock successfully. The working ROCm environment and
frontend dependencies were left intact.

The required Ruff gate and full native CPU suite passed: **5,399 tests,
43 skips and 11,414 subtests**, in 562.06 seconds. The complete client
`npm run check` passed. Its 108-file production build still matches every
backend bundle byte, so no generated bundle rewrite was needed. Existing
source-ledger builders reproduced all five reviewed ledgers without differences;
no catalog or qualification status was promoted. Gallery repair regressions
passed 88 tests, including changed recipes, tasks, inputs, sources, substituted
outputs and collection order. The fresh Gallery failures and remaining hardware
boundaries are recorded below; these CPU gates do not close them.

The full client browser gate also completed: **three shared-control cases** in
14.8 seconds and **255 mocked Studio cases** in 22.4 minutes, with no failures or
skips. It covers authoring, model/task changes, manual wiring, Auto submission,
durable status/recovery, template import/export and ordered output navigation.
Mocked backend responses do not establish model generation or GPU memory fit.
The revised split CI still requires a completed remote run after publication.

### Full-model and workflow evidence

| Machine/workload                                             | Expected result                                                                                               | Recorded result and limit                                                                                                                                                                                                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Local Radeon 8060S/gfx1151 shared-memory ROCm image campaign | Original recipe preserved; matched old/new ordered outputs; actual Auto/recompute behavior                    | Scoped parity and repairs described above; shared memory does not qualify dedicated 16 GB Windows VRAM                                                                                                                                                             |
| Local Qwen Control cold/warm Auto                            | Actual model reuse and fresh required execution without repeated loading                                      | Four original full runs; 144 ControlNet forwards, 288 base predictions, 144 scheduler updates; exact decoded repeatability; current-only repairs                                                                                                                   |
| MI300X Z-Image Turbo                                         | Full 1024-square BF16/no-quant 8-step recipe, preview and refresh                                             | Completed after the allocator correction; initial HIP failure retained                                                                                                                                                                                             |
| MI300X Qwen Image 2512                                       | Full 1024-square BF16/no-quant 50-step T2I, real seed change, preview/refresh                                 | Completed; this large-memory result does not reproduce the original low-free-RAM host                                                                                                                                                                              |
| Qwen T2I → img2img by Load Image/task adaptation             | Immediate readiness; same external image feeds required stages; authored strength consumed                    | Ordinary upload without refresh passed; authored 50 steps/strength 0.5 yielded 25 effective updates                                                                                                                                                                |
| Qwen img2img → Edit 2511 by model/task Apply                 | Preserve compatible edits/input; correct new model and required image wires; full execution                   | Full 40-step execution and UI recovery passed; inherited complex prompt output failed visual review                                                                                                                                                                |
| Independent Qwen producer → Edit consumer                    | Both actual owners and sinks, explicit typed media chain, full 50+40 steps                                    | Completed; peak Torch allocation about 111.83 GiB; consumer quality failed and smaller-GPU combined fit remains unproved                                                                                                                                           |
| Qwen Edit 2511 canonical single-reference                    | Requested full edit with original source/recipe                                                               | Full 40-step execution and separate visual inspection recorded; not interchangeable with the failed two-reference recipe                                                                                                                                           |
| Fresh stable 0.41 Qwen two-reference edit                    | Original prompts/ordered inputs, 40 steps, seed 5103, true CFG 4; usable output and refresh                   | One ordinary Auto Run completed with zero recorded browser errors; peak allocation 62,294,398,976 bytes; strong speckles and absent requested lavender sprig mean visual FAIL                                                                                      |
| Qwen two-reference one-space negative-prompt control         | Preserve the original positive conditioning and full recipe; isolate publisher-style negative-prompt behavior | Full genuine 40-step Auto execution and preview/reload completed; strong speckles and absent lavender remained: visual FAIL                                                                                                                                        |
| Same complex Qwen whole/native-math/vision/MM-token controls | Distinguish integration, attention and processor hypotheses                                                   | Full execution completed but artifacts/instruction failure remained; no demonstrated native-only defect or hardware cause                                                                                                                                          |
| Qwen Layered                                                 | Full 30-step, three ordered 640-square RGBA outputs with alpha/media identity                                 | Original and separately labelled substituted-source runs completed; collection delivery/navigation/refresh passed. Retained original layers pass bounded object separation, but opaque shoe-shaped remnants/shadows fail strict clean hidden-background acceptance |
| Active refresh and Stop/retry                                | Same task survives refresh; Stop reaches terminal cancellation before fresh retry                             | Scoped real passes; one worker-replacement journey retained connection diagnostics; not universal zero-error recovery                                                                                                                                              |
| Manual wiring, Undo/Redo, Save/reopen                        | Same authored graph/settings persist and later Run executes it                                                | Contract/mocked coverage and actual prepare-only journeys exist; not every saved/reopened modification received a full GPU Run                                                                                                                                     |

### Large checkpoints and access boundaries

| Model                    | Expected checkpoint                                                                 | Actual outcome                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| LLaDA2.1 Flash           | Load original 205.8 GB BF16 weights and exercise native Diffusers                   | Standalone **text** smoke completed one 32-token block/32 forwards, 28 GPU decoder layers plus four CPU-offloaded layers; peak Torch GPU allocation 179.77 GB and process RSS 199.95 GB; answer truncated before its result; no MoDiff/image/Auto qualification                                                                                                                                                                                                                                                                |
| Cosmos3-Super-Text2Image | Full 131.3 GB active image selection, mandatory guardrails, original 50-step recipe | Original generation weights and exact mandatory guardrail artifacts downloaded/verified; legitimate access and ordinary runtime Install/Activate passed. The first accepted attempt failed before weights at the documented Hub shared-blob bridge; backend `49d1f5b` corrected that boundary. The subsequent original 50-step native run loaded, encoded and denoised, then failed in Decode because the partial SDK stage lacked its mandatory safety configuration. Zero image outputs were delivered and no seed repeat ran. The reviewed adapter correction passed genuine SDK CPU regressions. On the final `3b7cab8`/`6d54bf8` pair, ordinary Template Create and both original configured 50-step Runs completed: cold seed 1143 in 133.67 seconds and resident seed 1144 in 35.95 seconds. Both original 1024-square images passed independent bounded publisher-scene review; actual task-bound events confirm mandatory AfterDecode succeeded in both. The second run reused the loader and encoder while denoising, decoding, AfterDecode and Preview actually recomputed. This validates those Custom attempts, without changing the catalog's unqualified status or promoting Auto/Gallery                                                                                                                                                                                                                                                                          |
| FLUX.2-dev               | Exact admitted full image route and complete model access                           | Original 112.8 GB required checkpoint downloaded/verified. Whole and native stages each completed cold and genuine changed-seed runs: 50 steps, 1024-square, BF16, guidance 4, no quantization/offload. All four original outputs pass bounded composition/text review. Encoder sequence length is whole-authored 512/native SDK-default 512. Whole/native same-seed RGB MAE is 10.1297/255 and 4.20973/255 for the two seeds: visually similar, not pixel exact. On the final `3b7cab8`/`6d54bf8` pair, the new ordinary template completed Create and its unchanged configured 50-step cold Run in 113.85 seconds. Its original PNG is byte-for-byte identical to the earlier accepted native seed-20260905 output, and independent visual review passed. Editing, Auto and Gallery remain separate |
| MiniMax H3               | One selected full 144 GB audio/video workflow                                       | Not downloaded/executed; publisher territorial eligibility must be resolved for the compute environment                                                                                                                                                                                                                                                                                                                                                                                                                        |

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

## Follow-up fixes and remaining issues

### P1: fresh-clone instructions obtain older code

**Implemented and contract checked:** both READMEs, client setup links and the
Windows fashion guide select the paired reviewed branch. Native examples retain
the same accelerator extra and correct platform executable paths. Actual fresh
GPU installation on Windows remains a separate check.

Relevant source: [backend README](../README.md), client README/setup links, and
[Windows fashion guide](windows-fashion-demo.md). Default branches were still
older `main` heads when reviewed; no release merge was performed by this handoff.

### P1: incomplete combined Auto working-memory authority

**Diagnosed guard implemented and CPU checked:** the existing planner now
withholds combined-fit authority for unknown independent owners, selects safe
existing release lifetimes and checks actual free memory before the next loader.
Unknown overlap requires Custom or reviewed working budgets. It does not convert
machine-capacity tiers into additional allocation charges.

Focused regressions cover distinct owners, reuse/shared identity, unknown demand,
unsafe overlap and primary-loader pre-allocation checks. Remaining work is
measured per-recipe working/load/activation budgets and constrained-device
hardware release/ownership acceptance. The mixed ComponentsManager boundary,
borrowed placement, compatible group hooks and shared-storage credit are now
diagnosed and CPU checked as described in point 10. Preserve full creative
settings and the corrected capacity/free distinction.

The executable resource inventory contains 37 declarations and 80 declared Auto
profile/mode pairs, with no complete explicit working-memory budgets. A retained
memory ledger separates nine task windows: three MI300X Custom image runs and
six local Radeon Auto Qwen runs. Their loading, resident storage and sampled
inference peaks remain observations tied to their original recipes and sources;
they were not converted into production budgets or new qualification claims.

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

**Offline reconciliation:** the 70 retained examples comprise 14 consistent
published Gallery contracts, 51 historical recipes, three entries with missing
proof/media and two older audio outputs that do not meet current duration
contracts. The 14 consistent contracts are not fresh latest-worker, Auto or
Windows runs. No exact real metadata repair was established; all 66 saved errors
remain unresolved. Ten current public templates have no accepted public example.

The maintainer repair tool previously attached current creator settings and
hashes to an older executed graph without checking equivalence. It now requires
retained original template authority, consumed settings, canonical graph, exact
task/node receipts and pinned input identities before rewriting sidecars.
Original source/runtime/model identities, capture time and ordered output byte
identities are preserved, with immutable before-copies and a rollback boundary.
Changed recipes, substituted media or missing
authority fail before writes; no retained real evidence was rewritten during
this review. See the client [Gallery asset guide](https://github.com/sdevil7th/MoDiff-client/blob/fix-ui-ux-issues/docs/template-gallery-assets.md).

A fresh offline verification on 9 October reproduced the same 66 errors over
70 examples. Strict current-template coverage reported 14 asset-backed templates,
65 uncovered templates and one user-supplied template, with five missing video
motion previews. This coverage gate is stricter than the count of ten templates
without any public example: historical examples do not supply current-recipe
authority. Bounded recovery searches did not recover the two legacy Qwen runs'
complete original model sets; the missing evidence and Ghibli hold remain open.

### P2: optional-runtime polling and staged activation

**Implemented and browser checked:** visible bounded polling separates abort,
transient connection loss and permanent/protocol failure, retains durable identity
and permits read-only status recovery. Mocked rendered tests cover 404, malformed
response, identity mismatch, stale timestamps, non-JSON 503 cutover recovery and
no duplicate mutations. Actual install, same-job refresh and replacement-worker
activation passed on MI300X. The last known progress survived transient network
loss and cleared when the same operation became ready. Safety-model inference
and generation remain independent acceptance checks.

### P2: harmless empty UI state invalidated an authoring preview

A genuine mocked-browser failure exposed a race between an absent `uiState`
and an empty plain object added while the user reviewed a model/task change.
The shared comparison now treats only those two empty representations equally.
Authored values, nonempty UI metadata, bindings, wiring and layout still
invalidate stale transactions. Undo/Redo and sibling metadata remain covered;
the complete 254-case Studio suite passed after the fix.

### P2: Windows safety-artifact verification

The inspected Windows CPU run passed ordinary uv setup, dependency checks,
preflight, tiny generation/LoRA and HTTP smokes, but failed one safety-artifact
test among 5,323 passing tests. The alias check compared access time as well as
file identity. Reading an unchanged regular file can update its access time,
which must not be interpreted as content replacement.

**Implemented and pushed:** `cc6f54eea82896adf6607735ca6b204625ce3113` excludes
access time from the alias mutation check while preserving mode, device/inode,
size, modification/change timestamps, resolved target and full digest checks.
Controlled tests reject real identity/metadata changes and accept an access-time
change. The native Linux CPU gate passed with 5,327 tests, 43 skips and 11,392
subtests. The subsequent Windows CI runs still failed regular-file safety
verification: the PR run had 5,337 passes and one failure; the push run had
5,336 passes and two failures, with 32 skips each. Both still failed the
access-time regression, and the push also failed the content-versus-storage-name
case. The first correction was therefore insufficient on Windows. These failed
runs remain retained; the second correction below preserves the identity,
alias-resolution and full-content checks while addressing the exact timestamp
comparison defect.

**Second correction verified on Windows:** `4432062b1ce64f77e2a804e7ae2c4a30c234e97e`
compares open-file metadata with open-file metadata, and pathname metadata with
pathname metadata. CPython 3.12 on Windows reports creation time through pathname
`stat` and change time through descriptor `fstat`; comparing those timestamp
families directly could reject unchanged bytes. The correction retains both
mutation checks, file identity, alias resolution and the full content digest.
The respective conversions are visible in CPython's
[pathname implementation](https://github.com/python/cpython/blob/v3.12.10/Modules/posixmodule.c)
and [descriptor implementation](https://github.com/python/cpython/blob/v3.12.10/Python/fileutils.c).
The new regressions also reject target-path changes that an unchanged snapshot
alias alone would miss. Both the [push CI run](https://github.com/sdevil7th/MoDiff/actions/runs/37915550014)
and [PR CI run](https://github.com/sdevil7th/MoDiff/actions/runs/37915557820)
passed on Windows, Linux and macOS. Each Windows run passed 5,388 tests and
11,391 subtests with 32 skips. This closes the reproduced CPU verifier defect;
actual Windows GPU validation remains separate.

A separate earlier Windows run crashed inside a native Torch BF16 Linear
operation with SIGILL; that test passed in the completed run and neither of the
two subsequent Windows runs crashed there. The failed record remains retained;
later passes do not establish the original crash's cause.

### P2: original Cosmos decode safety configuration

The first full native retry on backend `49d1f5b` and client `3e4f49c` passed the
corrected Hugging Face shared-blob boundary. It loaded the original Super model,
encoded the unchanged publisher caption and completed all 50 denoising steps at
1024-square, BF16, CFG 4, fixed seed 1143 and no offload or quantization. Decode
then failed because the partial Diffusers stage lacked
`config.enable_safety_checker`; its `requires_safety_checker` property raised an
attribute error. This was not an OOM. The task delivered zero images, and the
changed-seed repeat did not run.

**Reviewed correction:** the existing owned safety adapter registers the
mandatory `True` configuration before enabling and attaching the verified
checker. This preserves mandatory safety validation and the real AfterDecode
dependency. Four genuine stable-Diffusers partial Decode/AfterDecode regressions
failed before the change and passed after it, covering missing and false
configuration. The focused Linux CPU gate passed 157 tests and four subtests,
with 15 existing skips. The final `3b7cab8`/`6d54bf8` pair then passed actual
ordinary Template Create and two configured 50-step GPU Runs: seed 1143 cold and seed
1144 with resident weights. Real task-bound events confirm Decode, mandatory
AfterDecode and Preview succeeded for both, with newly produced images. Both
1024-square original PNGs passed independent review of coherent hands, wet-gray
clay vase, pottery wheel and warm studio lighting. The second Run reused its
unchanged loader/encoder and recomputed the required downstream stages. The
complete archive contains 122 verified payload files, including task, original
media, runtime, graph, browser and memory evidence; both owned samplers were
retired after completed tasks and an idle queue. Preserve the earlier failed
task and zero-output receipt. These Custom runs do not qualify Auto, other
recipes, Windows GPU execution or public Gallery media.

The receipts bind the unchanged configured 50-step SDK recipe and completed
stages; the retained browser messages do not independently count every model
forward. No extra execution-count claim is inferred from the progress label.

After both original images and all execution evidence were archived and
verified, one ordinary cache action released only the Cosmos model owner.
The fresh cache state confirmed its absence, with an idle queue and unchanged
worker/source. Torch allocation fell from 129,698,521,088 to 213,909,504 bytes,
and reservation from 135,226,458,112 to 213,909,504 bytes. This demonstrates that
scoped release on the recorded large GPU; it does not establish constrained
multi-model Auto scheduling.

### Original full FLUX template parity and the next editing check

On the same final pair, ordinary FLUX Template Create and the unchanged
configured 50-step cold Run completed in 113.85 seconds. All five task-bound
stage events report successful fresh execution. The original 1024-square PNG
is byte-for-byte identical to the earlier accepted native seed-20260905 output;
independent review confirms the blue teapot, readable **MORNING LIGHT** text,
plant and warm window scene. Raw and durable delivery and the visible Preview
are correlated to the new task. All 70 archive payload files matched their
size/content manifest; the sole sampler was retired after completion and idle.

The next large-GPU check is reference editing through the ordinary UI: import
this accepted current graph, attach **Load Image**, use its actual producer PNG
and request sage-green glaze while preserving the composition and text. Keep
the original 50 steps, 1024-square, seed, guidance and precision, then verify the
actual consumed reference, same-owner reuse, fresh downstream execution and
requested visual change. Source-only test preparation received independent
review and 33 CPU lineage/artifact checks, but no editing Run was submitted in
this block. Ordered multiple references and model replacement remain further
separate cases. Do not turn that preparation into an editing or Auto pass.

### P2: complex Qwen output quality remains unacceptable

The original output and controls remain retained. A full 40-step single-variable
publisher-style control changed only the negative prompt to one ASCII space,
preserving positive prompt, ordered references, seed, CFG and precision. It still
produced strong speckles and omitted the requested lavender. This is another
recorded visual failure; no native-only or hardware cause has been established.
Further work needs a matched upstream known-good reference or a reproducible
conditioning hypothesis. Avoid speculative global defaults or claims that the
processor bridge fixes quality.

### P3: test-browser prerequisite ordering

**Implemented:** client README/Windows guidance and the backend setup link now
put Chromium installation before `npm run check`, since request/proxy unit tests
also use it. `npm ci` itself passed on AMD; its initial quality failure was missing
Chromium. Fresh Linux validation also documents Playwright's browser system
dependencies. CI/CONTRIBUTING already used this ordering.

### Remaining qualification work

- Review actual Windows installation/GPU receipts after source integration; CI
  cannot replace that review.
- Complete route-specific cold/warm/recompute, prompt change, model-return,
  saved/reopened Run, full-weight LoRA replacement and resource-limited recovery
  only where existing evidence is incomplete.
- Preserve the completed real optional Install → verify → Activate → restart
  receipts and the accepted Cosmos cold/repeat generation; qualify other
  untested safety-enabled routes separately. Base
  Transformers/PEFT need none of those activation steps.
- Legitimate FLUX.2 and Cosmos guardrail access is now available. The exact full
  FLUX.2 runtime closure and both guardrail selections are downloaded and verified.
  Cosmos's normal sealed optional-runtime Install/Activate and subsequent
  ordinary-template cold/changed-seed safety-enabled generation passed. Preserve
  the old native zero-output failure and the final pair's separate accepted
  receipts; Auto, further workflows and Gallery admission remain distinct.
- Refresh exact-template/current-runtime evidence only for affected routes;
  preserve prior completed comparisons and failed attempts.

The setup-reference, optional-job diagnostics and combined-owner guard batches
are implemented. Gallery evidence reconciliation can reuse trustworthy retained
results; missing output/quality/hardware checks remain bounded, separate work.

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
