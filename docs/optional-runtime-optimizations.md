# Optional runtime optimizations

Last reviewed: 2026-08-10

MoDiff treats accelerator extensions and optional model libraries as reviewed
runtime contracts, not as uncontrolled additions to the main Python
environment. The current backend includes the fail-closed control and
artifact-validation scaffold for an app-owned staged overlay, but no package
profile is qualified for installation or activation yet. Existing hashless
optimization overlays are classified as `legacy_unqualified`; they are never
loaded or activated, and an explicit rollback deactivates them to the base
environment.

The first optional model-library contract moves Transformers `5.14.1` and PEFT
`0.20.0` together with their eight overlay-owned transitive distributions. It
is published as `candidate_unqualified` with `cutoverReady: false`, empty
artifact locks, and unavailable install/activation actions. Merely finding the
requested versions in the base environment does not make the contract
runnable.

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
rollback contracts so a future Setup surface does not have to invent a second
lifecycle. Current package install and activation requests return a fail-closed
conflict before a lease, job, network request, staged directory, or subprocess
is created. Runtime-only optimization features may still be explicitly
selected or qualified when their existing capability contract permits it.

## First-use execution boundary

Optional-runtime dependency metadata is intentionally separate from executable
delivery. Every current Diffusers execution profile is `base` delivered even
though it declares the composite Transformers/PEFT profile, so current missing,
wrong-version, or unqualified observations do not change browsing, Auto,
capability readiness, graph execution, or field actions. A future atomic
cutover changes an exact execution profile to `optional_overlay`; only then is
the optional runtime externally required for that execution.

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

A base-delivered `base_satisfied` requirement is execution-ready. When
`requiredNow` is true, only `state: active` is ready. Active requires the
current worker and status catalog to agree on the active overlay, and each
required profile must have `contractState: qualified` and
`cutoverReady: true`. The current `candidate_unqualified` profile therefore
cannot become runnable. Graph admission inspects only loader nodes on
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

The staged-overlay implementation is intentionally not an executable product
claim. Qualification still requires, at minimum:

- reviewed wheel filenames and SHA-256 values for the complete ten-package
  Transformers/PEFT closure on every supported Python/platform combination;
- a reviewed per-platform installer executable and immutable installer digest;
- Windows Job Object containment for breakaway descendants and handle-relative
  promotion/cleanup that is safe against directory replacement;
- a durable promotion commit/reconciliation record and stricter fresh-process
  side-effect containment; and
- clean-base, staged, restart/rollback, and representative no-weight/live
  workload evidence on the target machine.

Until those gates are closed, `installActionAvailable`,
`activationAvailable`, and `cutoverReady` remain false. The compatibility
routes do not make a candidate eligible by themselves.

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
