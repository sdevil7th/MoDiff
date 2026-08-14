# Optional runtime optimizations

Last reviewed: 2026-08-12

MoDiff treats accelerator extensions and optional model libraries as reviewed
runtime contracts, not as uncontrolled additions to the main Python
environment. The backend includes a fail-closed, artifact-locked app-owned
staged overlay with explicit per-platform qualification and delivery. Existing hashless
optimization overlays are classified as `legacy_unqualified`; they are never
loaded or activated, and an explicit rollback deactivates them to the base
environment.

The first optional model-library contract moves Transformers `5.14.1` and PEFT
`0.20.0` together with their eight overlay-owned transitive distributions. It
publishes one immutable six-target contract. Linux and Windows x86-64 are
`qualified` with install/activation actions available; Linux ARM64, Windows
ARM64, and both macOS architectures remain `candidate_unqualified` and
base-delivered.
The lock covers ten wheels on Python 3.12 for Linux, macOS, and Windows on
x86-64 and ARM64. Merely finding the requested versions—or merely publishing
these locks—does not make the contract runnable.

## Product contract

1. Setup and the runtime API show reviewed contracts and their qualification
   state; they must not imply that an unavailable package action can run.
2. Read-only discovery, workflow browsing/opening, Auto planning, capability
   inspection, and optional-runtime status never import or install an optional
   distribution.
3. A future qualified install must use a complete source-controlled wheel set,
   exact hashes, an authenticated installer, and an isolated staged directory.
   It may never resolve another copy of a base-owned dependency such as Torch.
4. Staging never mutates the active interpreter. Failed, cancelled, stale, or
   forged state remains non-active and leaves graph execution fail-closed.
5. A successful import/capability probe only proves that the feature can load.
   It never authorizes Auto.
6. Auto may select an optimization only after:
   - the user explicitly enables the capability;
   - the exact runtime, model artifact, mode, and result-affecting workload
     match a receipt;
   - an unchanged baseline exists;
   - the optimized run improves elapsed time or peak accelerator allocation by
     at least 2%; and
   - the user reviews and accepts the output.
7. Qualified choices are combined only when each choice has its own matching
   receipt. A package or feature update changes the runtime fingerprint and
   invalidates the old Auto eligibility.

The backend publishes bounded status, job, cancellation, activation, and
rollback contracts. Setup implements that lifecycle behind the backend-owned
`contractState: qualified`, `cutoverReady`, and per-action availability flags:
install or repair requires explicit consent, polls only the returned bounded job
identity, supports cancellation, and keeps activation and rollback behind their
own consent steps. Ambiguous staged environments do not expose activation.
The status catalog includes the effective `platform` and `machine`, and the
client names that target beside the runtime state. A pending target renders no
package controls and rejects direct install or activation before a lease, job,
network request, staged directory, or subprocess exists. A qualified target
requires explicit consent for install, activation, and rollback. Runtime-only optimization
features may still be explicitly selected or qualified when their existing
capability contract permits it.

## First-use execution boundary

Optional-runtime dependency metadata is intentionally separate from executable
delivery. Every current Diffusers execution profile publishes a complete
platform-delivery table: Linux and Windows x86-64 use `optional_overlay`, while
pending architectures use `base`. The effective target is resolved from that
reviewed table rather than a hidden platform shortcut. Discovery, workflow
browsing/opening, contract preview, and Auto planning remain non-installing on
every target.

Auto plans, workflow listings, model capabilities, and execution profiles
publish the same seven-field `optionalRuntimeRequirement` contract:
`schemaVersion`, `delivery`, `requiredNow`, `profileIds`,
`executionProfileIds`, `state`, and `reason`. The schema version is `1`; both ID
lists contain at most 32 unique bounded IDs. A no-contract base item may use
empty lists, while an overlay requirement must identify at least one optional
runtime and one execution profile. States are `base_satisfied`, `missing`,
`wrong_version`, `present_unqualified`, `staged`, `active`,
`busy_recovery_only`, `restart_required`, `repair_required`, and `unavailable`.
Malformed or ambiguous contracts and status catalogs fail closed as
`unavailable`.

A base-delivered `base_satisfied` requirement is execution-ready. On qualified
Linux/Windows x86-64 targets, when `requiredNow` is true, only `state: active`
is ready. Active requires the
current worker and status catalog to agree on the active overlay, and each
required profile must have `contractState: qualified` and
`cutoverReady: true`. An unqualified target cannot become overlay-runnable and
remains base-delivered. Graph admission inspects only loader nodes on
executable paths; field actions use their authorized module, action, and
values. Both are rechecked at the worker/pre-import boundary, and client
`runtimeHints` cannot authorize execution.

A non-active required overlay returns HTTP `409` with a bounded public blocker:
`error`, `category: optional_runtime`, `error_code`, `message`,
`recovery_hint`, and the seven-field `optionalRuntimeRequirement`. Worker
failures use the same redacted contract without tracebacks, local paths,
process details, or loader diagnostics. A live runtime mutation gate continues
to serialize every graph and field action. Persistent recovery or restart state
blocks only overlay-required execution; base-delivered work remains runnable,
including after an unsupervised activation records `restart_required` and
releases its completed mutation gate.

## Reviewed package pins

| Capability | Reviewed version | Current package action | Auto eligibility |
| --- | ---: | --- | --- |
| Hugging Face Hub kernels | `kernels 0.16.0` | Unavailable pending artifact locks | Exact qualified workload only |
| FlashAttention 2 | `flash-attn 2.8.3.post1` | Unavailable pending artifact/build locks | Exact qualified workload only |
| TorchAO | `torchao 0.17.0` | Unavailable pending artifact locks | Exact qualified workload only |
| Optimum Quanto | `optimum-quanto 0.2.7` | Unavailable pending artifact locks | Exact qualified workload only |
| bitsandbytes | `bitsandbytes 0.50.0` | Unavailable pending artifact locks | Exact qualified workload only |
| SageAttention | `sageattention 1.0.6` | Unavailable pending artifact locks | Manual experiment; never Auto |
| xFormers | `0.0.32.post2` for the Torch 2.8 CUDA profile | Base NVIDIA profile | Exact qualified workload only |
| AMD AITER | No universal pin | No generic installer | Manual, qualified Instinct/ABI combinations only |

xFormers releases are tied to a specific PyTorch ABI. MoDiff therefore pins
the Torch-2.8-compatible release rather than resolving the newest xFormers
package. FlashAttention is source-built because upstream does not publish one
wheel that safely covers every supported MoDiff CUDA/ROCm combination.

AITER is not presented as a one-click install on general AMD systems. Its
published builds target specific ROCm, Torch, and Instinct combinations.
Showing a generic install action would risk replacing the managed Torch ABI.
Setup links to the official build instructions for an administrator evaluating
a qualified deployment.

## Staged-overlay qualification boundary

Qualification is target-specific. Windows x86-64 has clean-base, staged,
supervised lifecycle, no-weight, and guarded live-model evidence. Linux x86-64
has clean-base, staged, supervised lifecycle, and no-weight evidence. Those two
targets are actionable. The other four target records remain pending and
base-delivered; their artifact locks alone do not make them eligible.

### Portable target qualification

`scripts/qualify_optional_runtime.py` prepares the same bounded qualification
on each supported operating-system/architecture pair without exposing a
product API or changing the source-controlled target table. It accepts an
already-qualified target or projects only the current pending target in memory. Preflight is
offline and non-mutating apart from a disposable copy of the already verified
managed uv executable:

```bash
./scripts/with-runtime-env.sh ./.venv/bin/python \
  scripts/qualify_optional_runtime.py --preflight-only
```

On Windows, the equivalent inspection command is:

```powershell
.\.venv\Scripts\python.exe scripts\qualify_optional_runtime.py --preflight-only
```

The command must report `status: ready` before the networked run. In particular,
Python must be 3.12, the managed uv receipt and executable must match the exact
source-controlled platform lock, and all ten staged distributions must be
absent from the base interpreter. Run from a dedicated prospective-base
checkout; do not uninstall packages from a normal development environment and
do not weaken the clean-base check. Keep any prospective dependency diff with
the evidence so the tested source state is reviewable.

The explicit-consent run downloads only the selected immutable wheel set into
a new temporary managed root. It uses the production install, validation,
promotion, activation, and rollback code; runs an offline tiny CLIP+LoRA
Transformers/PEFT workload in a fresh child; then starts another fresh child to
prove that rollback returned to a base process where every staged distribution
is absent. Evidence is bounded, excludes local paths, refuses to overwrite an
existing file, and should be written outside the repository:

```bash
./scripts/with-runtime-env.sh ./.venv/bin/python \
  scripts/qualify_optional_runtime.py --consent \
  --evidence ../modiff-optional-runtime-linux-x86_64.json
```

Use the same command on macOS and on each reviewed architecture. A passing JSON
file proves only the locked temporary overlay, fresh-process no-weight workload,
and rollback boundary on that exact host/source revision. It does not prove a
supervised HTTP restart/cancel/repair sequence, accelerator execution, a live
model/media result, or another platform. Run and record those remaining target
checks separately before changing any production action or cutover flag.

Linux x86-64 now has the portable and supervised evidence recorded below. No
local macOS qualification host is currently available, so macOS remains an
explicit pending target. Its evidence must come from the manual reviewed hosted
workflow or a contributor-controlled Mac. Until then, its checked contract
keeps both direct base dependencies and `delivery: base`; a passing reviewed
run can promote only that target in a later commit.

Qualification preparation now includes exact filename, URL, SHA-256, and size
locks for all sixty platform-wheel records, plus one immutable uv `0.11.26`
archive/executable pair for each supported target. The base installer writes a
receipt only after rehashing the reviewed executable, and the overlay path
rehashes it independently. Archive validation requires one matching METADATA,
WHEEL, and complete unique RECORD; every non-RECORD row must carry the exact
SHA-256 and size of its archived file, and RECORD must cover the archive exactly.
On Windows, the watchdog enters a non-breakaway Job Object with kill-on-close
before launching the installer, so cancellation or parent death contains the
entire descendant tree. These controls remain dormant while action flags are
false.

Promotion and cleanup now operate on exact filesystem objects rather than
resolved path strings. Windows renames the held source handle with
`FileRenameInfoEx`, write-through, and no replacement; cleanup quarantines the
same held directory and disposes each descendant by handle. Linux requires
`renameat2(RENAME_NOREPLACE)` relative to opened parents, while macOS requires
`renameatx_np(RENAME_EXCL)`; an unavailable exclusive primitive fails closed.
The install lease binds the original staging directory identity, so a replaced
name cannot be promoted or cleaned as though it were the validated tree.

Before promotion, MoDiff durably records the exact environment ID and canonical
manifest/validation digests. A post-rename record is written only after the
destination is re-inspected against those digests. On the next locked startup,
install, activation, or rollback operation, an interrupted prepared record is
either completed from the still-valid staged directory, acknowledged against
the already-promoted exact directory, or left as an explicit repair condition;
malformed, missing, duplicated, or identity-mismatched states never downgrade
to an absent journal.

Windows x86-64 qualification on 2026-08-12 exercised the dormant future-state
path in an isolated temporary managed root: the reviewed uv executable installed
all ten locked wheels (16,930,199 archive bytes), authenticated 3,441 wheel
files, passed isolated import/symbol/origin validation, promoted with a cleared
journal, activated only in the temporary state, and loaded
`transformers==5.14.1` plus `peft==0.20.0` from the overlay in a second fresh
process before rolling back to base. The run exposed and closed two Windows uv
integration details: local hashes must be expressed as `name @ file://...`
requirements with a separate `--hash=sha256:...`, and `--link-mode copy` is
required so the authenticated overlay never shares hardlinks with uv's cache.
The path-bearing requirements document plus uv's bounded `.lock` and
`uv_cache.json` bookkeeping are removed before authentication/promotion. No
model artifact was downloaded or executed. At that checkpoint it was one
Windows qualification, not Linux/macOS/ARM64 or representative model-workload
evidence, so source action and cutover flags remained false.

A second isolated Windows x86-64 run on 2026-08-12 exercised a bounded local
no-weight workload through that same production install, validation, promotion,
activation, fresh-process, and rollback path. With every Hugging Face offline
flag enabled, the activated worker loaded Transformers `5.14.1` and PEFT
`0.20.0` from the artifact-anchored overlay, constructed a tiny local CLIP text
encoder, injected PEFT LoRA adapters into its query/value projections, and
completed a finite `[1, 4, 16]` forward result with four trainable adapter
parameters. Diffusers reported its PEFT backend active. The fresh worker then
rolled back to a base process with no active environment. The run downloaded no
model artifact and changed no source action flag. It closes the Windows
no-weight staged-workload check only; it is not a clean-base install, supervised
server restart, live model/media run, or evidence for another target platform.

The prospective clean-base Windows x86-64 matrix then installed the reviewed
NVIDIA backend from a detached checkout with Transformers and PEFT removed from
the project dependencies and from required preflight imports. The managed
installer produced a compatible 64-package CUDA base; all ten overlay
distributions were absent; preflight was ready; and registry discovery loaded
132 nodes without loading any staged distribution. Starting from that base, the
same ten locked wheels and 16,930,199-byte archive set passed validation,
promotion, activation, the finite CLIP+LoRA workload above in a fresh process,
and rollback to a process with no active environment. This proves the Windows
clean-base/staged-runtime dependency split. It still does not qualify a live
model artifact, supervised server restart/repair, or another platform, and the
source dependency/action/cutover declarations therefore remain unchanged.

A supervised HTTP lifecycle then ran from that prospective clean base with only
the future action/cutover flags enabled in the detached process. The real
install endpoint staged and validated the same ten-wheel, 16,930,199-byte
closure and retained a ready job bound to its exact environment, profile, and
spec digest. Activation returned `restarting: true`, replaced the base worker,
and the new worker reported an active overlay and Transformers `5.14.1`.
Rollback returned `restarting: true`, replaced the active worker again, restored
the base process status, and reported Transformers absent. The install exposed
and fixed an invalid keyword call at the worker-thread progress boundary; both
optional-runtime and legacy optimization installers now schedule a bound update
callback, with regressions for each path. The temporary supervisor, detached
checkout, staged environment, and diagnostics were removed and port 8088 was
free. This closes Windows supervised install/activation/restart/rollback only;
live cancellation/repair, live model/media execution, non-Windows execution,
and source cutover remain pending.

A second supervised clean-base run exercised cancellation and repair through
the production HTTP boundary. Cancellation during the installer subprocess
advanced the exact job through `cancelling` to `cancelled`, removed staging,
promoted no environment, and left the same Transformers-free worker running.
After a later validated activation, a qualification-only one-byte overlay drift
caused the next worker to import no optional package and report
`repair_required`. Reinstall created a separately validated replacement, while
activation correctly refused to replace the still-selected corrupt runtime
directly. Explicit rollback restarted to base; activation of the replacement
environment from the completed job receipt restarted into Transformers
`5.14.1`; a final rollback restarted to base with Transformers absent. Setup
uses that completed-job environment identity before catalog inference, so the
immediate repair activation stays bound to the exact new receipt. The detached
tree, both staged environments, jobs, and server processes were removed. This
closes Windows supervised cancellation/repair, but not live model/media or
non-Windows execution and not source cutover.

A third detached Windows x86-64 run exercised the complete future qualified
guard with a real model and media output. Model Manager downloaded the
Apache-2.0, safetensors-only, no-custom-code
`optimum-intel-internal-testing/tiny-random-qwen-image` snapshot at immutable
commit `ef73a0df0cb8ccfa00cc178ec528c6e681791a10`. The activated composite
overlay then supplied Transformers `5.14.1` to the existing generic Qwen image
loader; `LoadPipeline -> Generate -> Image.Save` completed one 64 by 64 CUDA
step and produced a non-uniform RGB PNG. After rollback to a Transformers-free
base worker, the same loader was rejected before queueing with HTTP 409
`optional_runtime_staged`. This run also closed an app-owned download mismatch:
`POST /hf_download` now accepts a bounded exact lowercase 40-character commit
`revision`, forwards it to the Hub snapshot operation, and permits concurrent
join only for the same revision and file selection. All qualification-only
state was removed. This closes the Windows guarded live-model/media check, not
non-Windows qualification or the production dependency/action/cutover gate.

Linux x86-64 CPU qualification on 2026-08-12 used a detached checkout at
`b30c6b12989272b7399be1ed8102a25070ba00d0` with one reviewable prospective
base diff: remove the direct Transformers and PEFT project dependencies. The
managed installer selected Python `3.12.13`, installed a compatible 61-package
CPU base, and retained the exact uv `0.11.26` receipt with archive SHA-256
`6426a73c3837e6e2483ee344cbc00f36394d179afcba6183cb77437e67db4af0` and
executable SHA-256
`29b90e884c384e1578ac37335521d807c192aa44d5a4a9b9f4690bb3850e179d`.
All ten staged distributions were absent and registry discovery loaded all 20
module groups and 133 nodes without importing Transformers. The exact offline
preflight command was:

```bash
./scripts/with-runtime-env.sh ./.venv/bin/python \
  scripts/qualify_optional_runtime.py --preflight-only
```

It reported `status: ready`, clean base, dormant source flags, 10 Linux wheels,
17,457,395 archive bytes, and artifact-plan digest
`sha256:7e05d4b1fdb31d36da05f8b101aaa3a9ca746c5088c00f30bd1cf6d49474dda3`.
The explicit-consent command was:

```bash
./scripts/with-runtime-env.sh ./.venv/bin/python \
  scripts/qualify_optional_runtime.py --consent \
  --evidence ../modiff-optional-runtime-linux-x86_64-b30c6b1.json
```

It passed locked installation, validation, promotion, activation, the fresh
offline finite `[1, 4, 16]` CLIP+LoRA workload with four trainable adapter
parameters, and rollback to a second clean-base process. The bounded path-free
evidence was 1,518 bytes with SHA-256
`a90c3c84826a55af615b62eda9b8ea83c356691330eabba52e54ad3b0b790cde`;
no managed state was retained by the portable run.

The same prospective base then exercised the real supervised HTTP lifecycle
with qualified/action flags changed only in the detached checkout. An explicit
consent install exposed `installing`, `validating`, and `ready`; a separate job
advanced through `cancelling` to `cancelled`; and activation replaced the base
worker with a fresh worker loading exact Transformers `5.14.1` and PEFT `0.20.0`
from module and metadata origins bound inside the promoted environment. A
same-size, one-byte content mutation caused the next clean worker to import no
optional package and publish `repair_required`. Reinstall produced a separate
validated environment, direct replacement activation was refused with HTTP
409, rollback restarted to base, activation from the exact completed-job
receipt restarted into the repaired overlay, and final rollback restarted to a
base worker with all ten staged distributions absent. No model, media, or
Gallery asset was downloaded. Both server ports, the detached checkout,
environments, jobs, and evidence file were removed after review. This closes
Linux x86-64 clean-base/no-weight and supervised lifecycle qualification, not
AMD GPU execution, a Linux live-model/media run, macOS qualification, or source
cutover. Production dependency, delivery, qualification, and action flags stay
unchanged.

The final Linux repository replay used the normal managed CPU environment and
the documented gates:

```bash
uvx --from ruff==0.12.7 ruff check . --select E9,F
uv pip check --python ./.venv/bin/python
./scripts/with-runtime-env.sh ./.venv/bin/python \
  -m modiff.preflight --json --check-port 8088 --fail-on-error
./scripts/with-runtime-env.sh ./.venv/bin/python -m pytest -q
bash -n install.sh run.sh scripts/with-runtime-env.sh
git diff --check
```

Ruff and shell/diff checks passed, all 75 installed packages were compatible,
preflight was ready with port 8088 free, and pytest passed 1,236 tests plus
2,091 subtests with three platform skips. The only warning was the existing
Diffusers `torch_dtype` deprecation. On sibling client commit `aded6ca`,
`npm ci`, `npm run check`, `npx playwright install chromium`,
`npm run check:ui`, `npm run build`, and `npm run bundle:check` passed: two
Linux shared-control visual tests and all 99 mocked Studio tests passed. The
26-file production mirror was byte-identical with aggregate manifest SHA-256
`bb10bdd69e6a91fa40d8d6a97c33229b9ea39b8efcc64352946e7b98c166f330`.
Entry `index.js` was 1,035,340 raw/283,322 gzip bytes with SHA-256
`23fe0bcd224eebc2c68fd09109cffe4d0b5ec57dce51e7cf730bfb818963ee40`;
the four JavaScript chunks totaled 523,108 gzip bytes against the 523,264-byte
cap. A fresh backend returned HTTP 200 for `/`, `/assets/index.js`, and
`/health`. No Gallery media or model asset was generated or downloaded.

`.github/workflows/qualify-optional-runtime-macos.yml` is the smallest pending
hosted macOS proposal: it is `workflow_dispatch` only, asserts the explicit
`macos-15` runner is ARM64, creates one exact two-dependency prospective-base
patch, verifies and applies that same patch, requires `status: ready`, runs the
same consented qualifier, and uploads the resulting bounded diff, evidence,
and target context for 14 days. A regression executes the patch check against
the current project rather than only inspecting workflow text. The workflow
has not run. As of 2026-08-14, the
[official GitHub-hosted runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
lists the standard `macos-15` label as an ARM64 M1 runner; the workflow still
checks `uname -m` before changing the prospective checkout so a future label
drift fails closed. This runner-catalog inspection is infrastructure metadata,
not macOS executable evidence or permission to enable any production flag.

## Runtime features

The following features have concrete runtime implementations and remain
disabled until explicitly selected or applied by an exact Auto receipt:

- Diffusers attention dispatcher backends, including native SDPA, xFormers,
  FlashAttention, Hub FlashAttention variants, SageAttention, and AITER when
  their capability probes pass.
- Diffusers regional compilation of repeated blocks.
- Diffusers denoiser caches with model/workload output review.
- Diffusers layerwise casting with float8 storage and an explicit compute
  dtype. The runtime prevents stacking incompatible casting hooks on a cached
  pipeline.
- Channels-last layout for explicitly selected convolutional UNet/VAE
  components.
- Existing model-specific quantization and offload recipes.

The following upstream features are deliberately visible but not enableable:

- generic quantization combined with offload;
- multi-GPU context parallelism; and
- generic fused QKV projection.

They require model- and ordering-specific execution contracts. A documentation
entry is not treated as proof that an arbitrary MoDiff graph can use the
feature safely.

## Official sources reviewed

- [Diffusers attention backends](https://huggingface.co/docs/diffusers/optimization/attention_backends)
- [Hugging Face kernels installation](https://huggingface.co/docs/kernels/main/installation)
- [FlashAttention repository and platform requirements](https://github.com/Dao-AILab/flash-attention)
- [TorchAO inference workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html)
- [Diffusers memory optimization](https://huggingface.co/docs/diffusers/optimization/memory)
- [Diffusers quantization API](https://huggingface.co/docs/diffusers/main/api/quantization)
- [Diffusers optimization CLI](https://huggingface.co/docs/diffusers/main/using-diffusers/cli)
- [xFormers releases](https://github.com/facebookresearch/xformers/releases)
- [AMD AITER](https://github.com/ROCm/aiter)
- [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/index)

These are reviewed pins. MoDiff does not resolve “latest” at runtime. A version
change requires a new source review, staged validation, and qualification
receipts.
