# AMD cloud image validation

Use the shared [three-machine application procedure](three-machine-app-validation.md)
for graph editing, custom nodes, task lifecycle, persistence and other media
families. This guide supplies the dedicated AMD image-preparation boundary.

Prepare and review the backend and client commits before provisioning. Record
both commit IDs and the built client asset hashes; use those exact reviewed
commits on the cloud machine. Clone into a fresh checkout, without credentials,
local installation state, virtualenvs, extension approvals, personal workflows,
model caches or generated outputs from an operator checkout. Retain required
curated contract files: excluding all of `data/` removes application contracts.
Keep SSH private keys outside the repositories. Verify the host fingerprint
through the provider console before connecting.

## Hardware and runtime checkpoint

Confirm the actual accelerator, partition, visible device count, dedicated
memory, host RAM, available disk, OS/kernel, device permissions and runtime.
An MI300X has 192 GB HBM3; an MI100 has 32 GB HBM2. A partition can expose less
capacity, and neither advertised capacity nor free memory proves a model fits.
[AMD MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html),
[AMD MI100](https://www.amd.com/en/products/accelerators/instinct/mi100.html).

The existing MoDiff `amd-instinct-rocm-linux` profile is a preview profile for
`gfx942`, with Torch `2.10.0+rocm7.14.0` and ROCm 7.14. The preparation script
requires Linux and exactly one visible dedicated device in that family.
It records observed capacity without assuming a full MI300X partition.
MI100/`gfx908` is outside this MoDiff profile, even though AMD publishes ROCm
support and PyTorch packages for that architecture. Stop for a separate profile
review if the supplied machine is an MI100; do not use architecture overrides,
install `gfx942` wheels on it or silently substitute a newer runtime.

On the actual cloud machine, use the
[direct Instinct uv setup and launch commands](developer-setup.md#instinct-sdk-setup-on-linux).
They install only into the fresh venv using the reviewed AMD requirements and
adjacent index configuration, then run package checks and ordinary preflight.
Receipt-free detection binds the actual single dedicated `gfx942` device and
installed SDK package versions; do not create installer state by hand or rely on
an environment profile selector. Verify the provider's actual kernel-mode driver
against [AMD's compatibility matrix](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/compatibility/compatibility-matrix.html)
for the app-local SDK, rather than equating its advertised userspace version with
driver compatibility. Keep host checks and the tiny tensor result in the evidence.

Normal Linux ROCm launch uses Torch's default allocator, without injected
expandable segments, for both Instinct and Ryzen. Explicit allocator environment
settings still take precedence. Some HIP virtual-memory allocation paths retain
file descriptors per segment and can fail with ample free GPU memory; capture
the worker's allocator environment and file-descriptor soft/hard limits alongside
memory observations. Do not diagnose that failure as an Auto RAM/VRAM budget
rejection or change process limits automatically. Compare allocator settings only
in separately labeled fresh processes with the same unchanged recipe; a result
does not qualify another allocator policy or GPU.

The [guided installer](accelerator-installation.md#instinct-mi300x-cloud-preview)
is a separate alternative. Do not copy a Ryzen virtualenv, apply Ryzen driver remediation
or alter provider drivers speculatively. Use a dedicated non-root application
user and a loopback backend, reached through an SSH tunnel on a separate local
browser origin. Record the authorized test window and result-backup deadline;
job shutdown and provider instance destruction are separate actions.

## Portable preparation report

Use a dedicated run directory separate from source and the selected Hub cache.
In the fresh backend configuration, set every mutable `[paths]` entry
(`work_dir`, `data`, `images`, `videos`, `audio`, `models`, `upscalers`, `temp`)
to an absolute path under that directory. `app_root` remains the checkout.
Select the explicit Hub cache and offline mode for cached-artifact tests, with
no configured token. Stage only the selected immutable model revisions and
original input files needed for the first batch. Verify copied weight checksums
and ordinary app artifact readiness; a snapshot directory may be incomplete.
Existing gated access or terms must remain separate from technical testing.

Export the actual ordinary workflow, retaining its full original recipe. Put
the unchanged export and original image/mask/reference bytes under the private
run directory. From the backend checkout, run its selected environment's Python:

```sh
.venv/bin/python scripts/prepare_amd_image_validation.py \
  --run-root "$AMD_VALIDATION_ROOT" --cache-root "$AMD_HUB_CACHE" \
  --workflow "$AMD_VALIDATION_ROOT/workflow.json" \
  --snapshot "$MODEL_REPO@$MODEL_REVISION" \
  --input "$AMD_VALIDATION_ROOT/reference.png" \
  --output "$AMD_VALIDATION_ROOT/preparation.json"
```

Set these variables to the chosen absolute directories and exact published
repository/40-character commit. Repeat `--snapshot` for each base, adapter,
ControlNet, prior or upscaler selection; repeat `--input` in original binding
order, including masks. Omit inputs for a text-only workflow. The script hashes
workflow and input bytes, records snapshot file names/sizes, and reuses the
existing hardware/runtime validation, including its tiny device tensor probe.
It imports no operation registry or custom extensions and submits no graph,
loads no model, downloads no artifact and changes no configuration.

For a started, idle owned worker, append:

```sh
  --backend-url "http://127.0.0.1:$AMD_BACKEND_PORT" \
  --backend-pid "$AMD_BACKEND_PID" --backend-created "$AMD_BACKEND_CREATED"
```

Use the actual listener worker's PID and exact `psutil.Process(pid).create_time()`
value, rather than a supervisor or shell PID. This verifies listener ownership,
process-start source, ready runtime, empty current/queued tasks, source root,
private mutable storage, explicit cache and offline/no-token configuration using
GET requests only. It is a point-in-time check, not an execution lease or proof
that no other user occupies the physical GPU. Run it while the test GPU is idle;
ordinary dispatch must still revalidate the graph and resource plan.

The report is created once, owner-readable only, and existing evidence is never
overwritten. `preparationPassed` does **not** mean executable workflow validation,
complete artifacts, model fit, actual generation, Auto qualification or approved
visual quality. No backend options means no live worker ownership check. A
failed check writes no successful preparation report; preserve the error and
correct the actual setup before proceeding.

## Dedicated-device run sequence

Use ordinary Gallery/operation authoring and Run through the existing executor;
this preparation tool adds no alternative execution path. Start with a full
canonical short-step image recipe, then batch by already-cached model family:

1. Run cold in ordinary Auto mode. Capture its actual effective placement,
   original geometry/steps/seed/CFG, artifact pins, task receipt and every raw
   Preview item. Continue only after a real visible output and terminal task.
2. Repeat the same recipe warm without releasing models, then change the seed
   and positive prompt while preserving the original negative and other fields.
   Observe fresh forwards and re-encoding; a reused preview is insufficient.
3. Exercise owner-page refresh while active, then a separate ordinary Stop,
   terminal cancellation and full-recipe retry. Never start the retry before the
   cancelled task reaches terminal state or interrupt another owner's task.
4. Test resident (`none`) and reviewed offload policies separately using Custom
   only when admitted for the actual full graph. Preserve effective placement
   differences from Auto and any rejection. Do not lower geometry/steps,
   clamp ControlNet scale or disguise a placement change to force a comparison.
5. Expand to full edit/multi-reference, mask/outpaint, named LoRA, ControlNet and
   upscale recipes. Capture every sink/local media index, including generation
   and upscaled previews. Preserve unavailable or blocked recipes as such.

Use the existing `scripts/capture_runtime_memory.py` before model allocation
for the exact worker PID and correlate UTC samples with submission/terminal
events. It is bounded to one hour; rotate one writer at an idle boundary and
retain footer, next header and any gap. OS samples are lower bounds on peaks,
not GPU residency, resource budgets or qualification.

Follow [image-template validation](image-template-validation.md) for actual
consumed contracts and `scripts/compare_image_template_outputs.py` for all
ordered raw outputs. Compare old/current on the same actual cloud hardware and
runtime where feasible; hashes from another GPU are not a numerical baseline.
Retain requested-versus-consumed and resident-versus-offload differences, and
review full images against prompts/inputs separately from pixel comparison.
Cloud results do not qualify Windows, NVIDIA or local hardware.

Save receipts, source/asset identities, runtime facts and original raw output
hashes in private evidence. Back up and verify those bytes before any authorized
destructive turnover. Never delete operator caches, stop unowned processes or
promise provider-side teardown without its explicit authority.

## Execution and visual quality

Completed full BF16 Qwen Edit-2511 runs on a dedicated `gfx942` device establish
execution only for their recorded source, runtime, inputs and recipe. A matched
two-reference, 40-step comparison across native and official whole execution,
including separate math-attention and multimodal-token controls, still produced
strong image artifacts and missed the requested edit. These results do not
establish visual acceptance or show that either attention selection or the
processor compatibility bridge resolves that case.

Review canonical source images and edit instructions separately from complex
synthetic-reference cases. Preserve every ordered Layered RGBA output and inspect
its alpha as well as its color; a successful task, layer count or flattened
composite cannot establish decomposition quality. Keep execution, UI lifecycle,
resource observations, numerical comparison and visual acceptance as separate
claims. No result qualifies an untested recipe, platform or source revision.

## Large-model selection

The 2026-10-08 publisher-metadata review compares one executable checkpoint's
selected components. A repository can contain original and converted weights,
several task variants, and duplicate precision formats; its total size is not
one model's inference requirement. Weight-file bytes also do not establish RAM,
VRAM or activation peaks.

| Reviewed checkpoint | Selected full-precision weight files | Current test boundary |
| --- | ---: | --- |
| MiniMax H3, one converted audio/video workflow | 144,016,405,316 bytes (144.0 GB) | Metadata and existing source contracts reviewed; no model download or execution in this campaign. Territory eligibility must be resolved first. |
| Cosmos3-Super-Text2Image, five-component Omni text-to-image workflow | 131,299,159,064 bytes (131.3 GB) | Source/runtime integration and hardware/output qualification are separate requirements; no completed generation is claimed. |
| FLUX.2-dev, selected Diffusers component weights | Approximately 112.8 GB | Required gated access, complete artifacts and live execution remain pending. |

These three entries cover the reviewed image and audio/video workflows,
not an exhaustive ranking of every model on the Hub. No reviewed single
Diffusers image checkpoint in this audit contains 200 GB–1 TB of required
weights. MiniMax's whole repository contains approximately 498.3 GB of
safetensors, including original/converted task variants; the two converted
task selections share components and together contain approximately 210.3 GB.
Neither figure describes one 498.3 GB or 210.3 GB inference checkpoint.
[MiniMax Diffusers integration](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3).

The selected Cosmos snapshot at
`daf3d374804be4c512c2135568a7cb95d4341d79` contains 132,489,727,416 bytes
of safetensors. Its vision encoder contributes 1,190,568,352 bytes and is
not an executable component of the reviewed text-to-image Modular index.
The transformer, VAE and sound tokenizer account for the table's 131.3 GB.
Use that exact publisher index and example caption with the full BF16
50-step, CFG 4, 1024-square, seed-1143 recipe. The mandatory safety runtime
and its exact local artifacts must succeed before generation-model loading.
Do not replace the full model with a 4-step or quantized checkpoint for this
qualification.
[Publisher checkpoint](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image/tree/daf3d374804be4c512c2135568a7cb95d4341d79),
[Cosmos Diffusers integration](https://huggingface.co/docs/diffusers/main/api/pipelines/cosmos3).

MiniMax H3's published community license excludes the United States, United
Kingdom, European Union and Republic of Korea. Confirm the actual compute
region as well as the operator's eligibility before any model download or
execution; an ungated repository does not establish authorization. The current
campaign's observed compute region is an excluded territory, so this model
requires separate publisher authorization or a suitable, authorized environment.
[Exact publisher license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE).

### Broader native Diffusers survey

Stable Diffusers 0.41 also includes discrete diffusion **text** pipelines. This
matters when interpreting the request for a 200 GB–1 TB model: LLaDA2 Flash is
a real checkpoint above 200 GB, while its output and integration requirements
are different from MoDiff's image workflows.

| Additional reviewed native family | Selected weight-file bytes | Scope of evidence |
| --- | ---: | --- |
| LLaDA2.1-flash, discrete diffusion text | 205,782,452,128 bytes (205.8 GB; approximately 191.65 GiB), 32 shards | Full checkpoint downloaded and hash verified; standalone BF16 SDK smoke completed one 32-token block and 32 steps with CPU/GPU placement. No MoDiff workflow or answer-quality qualification. |
| LLaDA2.0-flash, discrete diffusion text | 205,782,433,824 bytes (205.8 GB), 42 shards | Same index-based selection; separate checkpoint, not combined with 2.1. |
| JoyAI-Echo, audio/video, with external Gemma 3 12B | 70,514,517,046 bytes (70.5 GB) | Native `EchoModularPipeline` index selects 46.1 GB in five Echo component folders plus 24.4 GB of gated Gemma weights; excludes standalone DMD, FP8 and FP4 duplicates. |
| Nucleus-Image, text-to-image | 51,633,577,862 bytes (51.6 GB) | Native `NucleusMoEImagePipeline` index selects transformer, text encoder and VAE. The older `NucleusMoE-Image` Hub name redirects to `Nucleus-Image`. |
| Ideogram 4, text-to-image | Unknown | Native `Ideogram4Pipeline` is present; public checkpoint metadata returned HTTP 401, so no checkpoint-size ranking is established. |

The LLaDA2.1 Flash snapshot is
`2bf95e86ade33c1adb5c1e223b7db2a76dc3bdd6`; its safetensors index selects
all 32 unique shard files once. The publisher describes a 100B non-embedding
MoE model; Hub metadata records approximately 102.9B total parameters, mostly
BF16 with a small FP32 subset. Its model card declares Apache-2.0. The older
2.0 Flash snapshot is `744c3f8c6c8317d2377d6d16d8a3d4be2caef563`.
[Exact 2.1 index](https://huggingface.co/inclusionAI/LLaDA2.1-flash/blob/2bf95e86ade33c1adb5c1e223b7db2a76dc3bdd6/model.safetensors.index.json),
[publisher model card](https://huggingface.co/inclusionAI/LLaDA2.1-flash/blob/2bf95e86ade33c1adb5c1e223b7db2a76dc3bdd6/README.md),
[exact 2.0 index](https://huggingface.co/inclusionAI/LLaDA2.0-flash/blob/744c3f8c6c8317d2377d6d16d8a3d4be2caef563/model.safetensors.index.json).

`LLaDA2Pipeline` and `BlockRefinementScheduler` are native stable Diffusers
implementations. Their threshold, token editing and post-refinement controls
cover the publisher's speed/quality modes. The official example loads a model
through Transformers with `trust_remote_code=True`; the Flash config maps to
publisher `LLaDA2MoeModelLM` Python code. MoDiff admission still requires a
separate reviewed model-loading implementation, text task/output contracts and
hardware/output qualification. Approximately 191.65 GiB of full weight files
requires placement headroom for runtime and activations; the measured SDK smoke
below used CPU offload.
[Native stable pipeline source](https://github.com/huggingface/diffusers/blob/086bf9578c0f4acbc66e48cd1e7cc26befd9e10f/src/diffusers/pipelines/llada2/pipeline_llada2.py),
[exact publisher class mapping](https://huggingface.co/inclusionAI/LLaDA2.1-flash/blob/2bf95e86ade33c1adb5c1e223b7db2a76dc3bdd6/config.json).

#### Standalone LLaDA2.1 Flash SDK smoke

On the droplet's MI300X VF (`gfx942`), the original pinned checkpoint completed a
standalone text smoke using native Diffusers `LLaDA2Pipeline` 0.41.0, Transformers
5.19.0, Accelerate 1.15.0 and Torch `2.10.0+rocm7.14.0`. All 39 checkpoint,
publisher code and tokenizer files, totaling 205,792,373,581 bytes, were verified
by SHA-256 before local-only loading. Model parameters used BF16 without
quantization; attention used SDPA and KV caching was disabled.

An explicit Accelerate device map was inferred from the empty model, keeping
complete decoder layers together. Its budgets were observed free GPU memory
minus 16 GiB and available host RAM minus 32 GiB. The resulting map placed the
embeddings and first 28 decoder layers on the GPU, with the final four decoder layers and
output-side modules assigned to CPU offload. There was no disk offload or
post-load whole-model device move. This measures that specific mixed placement;
it does not establish full GPU residency.

The smoke generated one 32-token block in exactly 32 successful forwards and
32 callbacks. Its original 35-token chat prompt was left-padded to 64 input
tokens with 29 padding positions whose attention mask was zero. The arithmetic
input was `Calculate 1+5-28*0.5-200=?`. It used greedy sampling,
temperature 0, seed 1143, threshold 1.1, minimum top-k 1, no editing, no
post-refinement and no early EOS stop. These are explicit short-smoke settings,
not a publisher quality preset.

The SDK receipt recorded 159.58 seconds total, including 104.64 seconds of model
loading and 46.60 seconds of generation. Torch reported peak allocated GPU
memory of 179,769,529,856 bytes (179.77 GB); process peak RSS was
199,948,599,296 bytes (199.95 GB). These distinct memory measurements are not
the checkpoint's 205.8 GB file size or an Auto resource budget. The timings are
observations from one run, not a throughput benchmark.

The returned text exhausted its 32-token limit before the final arithmetic
answer. This establishes a standalone BF16 text execution smoke only:
MoDiff graph execution, image generation, longer responses, answer quality and
Auto/Gallery qualification remain untested for this checkpoint. Cosmos's
mandatory gated safety artifacts and its separate image-model execution remain
pending; this text result grants no Cosmos or application admission.

Echo's audited index is
`jdopensource/JoyAI-Echo@d5a781ef08adb1f84748431ff5366de6e320d62d` and
its external Gemma snapshot is
`google/gemma-3-12b-it@96b6f1eccf38110c56df3a15bffe176da04bfd80`.
The publisher index leaves the external revision unpinned; this audit binds
that snapshot explicitly. Its `ltx2` connector/vocoder type hints resolve
to Diffusers' built-in LTX-2 pipeline namespace. Gated Gemma access, compatible
local components and hardware/output qualification remain separate checks.
[Exact Echo index](https://huggingface.co/jdopensource/JoyAI-Echo/blob/d5a781ef08adb1f84748431ff5366de6e320d62d/modular_model_index.json),
[Echo integration](https://huggingface.co/docs/diffusers/v0.41.0/api/pipelines/echo).
Nucleus uses `5e963db4fd0a65c7e4faf53ca2d4eca567c4dcfa`.
[Exact Nucleus index](https://huggingface.co/NucleusAI/Nucleus-Image/blob/5e963db4fd0a65c7e4faf53ca2d4eca567c4dcfa/model_index.json),
[Ideogram integration](https://huggingface.co/docs/diffusers/v0.41.0/api/pipelines/ideogram4).

The expanded survey also inspected native GLM-Image, FIBO, PRX Pixel,
HunyuanImage 2.1 and DiffusionGemma publisher metadata; their reviewed required
weights are smaller than the three image/audio-video entries above. The much
larger HunyuanImage **3.0** Transformers/custom-code checkpoint does not become
native Diffusers support through the similarly named **2.1** pipeline. No
reviewed native image checkpoint in this bounded survey supplies 200 GB–1 TB
of required weights; this remains a survey result, not a universal size limit.

Downloaded weights, CPU package qualification and a successful task each prove
only their recorded scope. Keep standard-runtime compatibility, mandatory
moderation, new-node workflow authoring, numerical/visual quality, repeated
execution and measured Auto resource qualification distinct in the report.

## References

- [AMD ROCm 7.14 compatibility matrix](https://rocm.docs.amd.com/en/docs-7.14.0/compatibility/compatibility-matrix.html)
- [AMD PyTorch packages](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
- [AMD Developer Cloud configuration and billing FAQ](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html)
