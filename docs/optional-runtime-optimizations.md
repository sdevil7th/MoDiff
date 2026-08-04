# Optional runtime optimizations

Last reviewed: 2026-07-29

MoDiff treats accelerator extensions as optional runtime capabilities, not as
uncontrolled additions to the main Python environment. An optional package is
installed into an app-owned staged overlay, validated in a fresh process
against the active Python, Torch, Diffusers, and accelerator profile, and only
then offered for activation. Activation requires a worker restart. The last
validated overlay remains available for rollback.

## Product contract

1. Setup shows packages and runtime features supported by the current managed
   profile.
2. Package installation never mutates the active interpreter. A failed build
   or import probe leaves the running environment unchanged.
3. ABI-sensitive packages are installed without resolving another copy of
   Torch. Source builds receive an app-local build toolchain.
4. A successful import/capability probe only proves that the feature can load.
   It never authorizes Auto.
5. Auto may select an optimization only after:
   - the user explicitly enables the capability;
   - the exact runtime, model artifact, mode, and result-affecting workload
     match a receipt;
   - an unchanged baseline exists;
   - the optimized run improves elapsed time or peak accelerator allocation by
     at least 2%; and
   - the user reviews and accepts the output.
6. Qualified choices are combined only when each choice has its own matching
   receipt. A package or feature update changes the runtime fingerprint and
   invalidates the old Auto eligibility.

The Setup panel exposes install progress, activation, rollback, opt-in,
compatibility probes, qualification actions, and the upstream documentation.

## Reviewed package pins

| Capability | Reviewed version | App-managed | Auto eligibility |
| --- | ---: | --- | --- |
| Hugging Face Hub kernels | `kernels 0.16.0` | NVIDIA/Linux | Exact qualified workload only |
| FlashAttention 2 | `flash-attn 2.8.3.post1` | CUDA/ROCm source build | Exact qualified workload only |
| TorchAO | `torchao 0.17.0` | Supported profiles | Exact qualified workload only |
| Optimum Quanto | `optimum-quanto 0.2.7` | Supported profiles | Exact qualified workload only |
| bitsandbytes | `bitsandbytes 0.50.0` | NVIDIA profiles | Exact qualified workload only |
| SageAttention | `sageattention 1.0.6` | NVIDIA/Linux | Manual experiment; never Auto |
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
