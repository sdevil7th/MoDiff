# Runtime support matrix

| Profile | Tier | Automated proof | Physical proof |
|---|---|---|---|
| NVIDIA CUDA (Windows/Ubuntu x64) | Supported | Manifest, resolver, CPU-host contract | Required for release |
| Apple MPS (Apple Silicon) | Supported installer | Manifest and contract | Required per model |
| Intel XPU (Linux/Windows x64) | Preview | Manifest, installer, and XPU tensor contract | Required per model/device/driver recipe |
| AMD ROCm Linux, Ubuntu 24.04.3, gfx1150/gfx1151 | Supported stack | Detector fixtures | MoDiff model proof required |
| AMD ROCm Linux, Ubuntu 26.04 | Experimental | Detector fixtures | Local tensor and model proof required |
| AMD Instinct MI300X, Ubuntu 24.04, gfx942 | Preview | Separate SDK profile, resolver and installer/runtime contracts | Cloud device tensor and frontend model proof still required |
| AMD PyTorch Windows | Conditional, install blocked | Official-platform manifest and explicit guidance | Complete MoDiff SDK wheel lock and physical model proof required |
| CPU | Supported | Install and tensor smoke | Reference host required |

“Supported” describes installation/runtime qualification, not model performance. `/model_capabilities` remains the source of graph capability. Exact model, dtype, placement, optimization, driver, and hardware qualification is receipt-specific; an unqualified recipe may be runnable with a warning but must not be described as optimized.

## Model families and support boundaries

The backend catalog and exact execution specifications determine available
routes. A family name does not authorize every upstream pipeline, arbitrary
composition, model switch, or resource configuration. Ordinary pipeline Blocks
contain ordinary nodes; only genuine Modular routes expose upstream hierarchy.

| Family | Implemented scope | Limits to retain when testing |
| --- | --- | --- |
| Qwen Image | Reviewed text-to-image, image-to-image, inpaint, ControlNet, Edit/Edit Plus, and Layered routes; typed Modular composition, iteration links, saved User Nodes, and explicit repairs | Fine geometry and material instructions can drift. Multi-reference editing is not pixel-exact preservation; RGBA layers do not promise perfect object isolation. ControlNet conditions must have the channels required by the selected encoder; Canny has an explicit RGB option while retaining its grayscale default. |
| FLUX | Native Modular FLUX.1, Kontext, FLUX.2, and Klein Base/Distilled routes; ordinary Fill, control, Redux, KV, and reviewed adapter paths | Dedicated Canny/Depth weights, separate ControlNets, ordinary Klein, and Klein KV have different contracts. Outpaint seams, masked details, colour adherence, and multi-reference dominance or duplicated subjects require visual review. Tiny attention/LoRA-scale effect tests do not establish full-weight native use with a loaded LoRA. |
| Audio | MiniMax Music3's native Modular hierarchy and ordinary ACE-Step, LongCat AudioDiT, and AudioLDM2 adapters | Music, sound, lyrics, variation, and editing controls are task-specific. LongCat duration follows upstream VAE-frame quantization; do not pad a result to claim an exact duration. AudioLDM2 exposes candidate waveforms through Audio Variations. Successful delivery and repeatable hashes do not establish listening, lyric, seam, or publication approval. |

Use [ordinary image node guidance](../modules/DiffusersImage/README.md) for
FLUX entry selection, latent decode, attention arguments, and adapter inputs.
Use the [Modular guide](../modules/ModularDiffusers/README.md) for editable
upstream compositions and the [API reference](api-reference.md#studio-preview-state)
for captured execution inputs and measured output metadata.

## Acceptance and remaining qualification

Keep one evidence row per exact route or template, with separate results for:

1. Registry and typed-contract compatibility against the pinned upstream source.
2. Native insertion, parameter edits, nested resize/expansion, wiring, Fix,
   Undo/Redo, Save/refresh, and User Node reinsertion.
3. Actual upstream component execution and adapter parity with small test models.
4. Full-weight frontend execution at the declared settings, including the edited
   composition, exact model revision, dtype, offload, seed, and captured inputs.
5. Independent repeat inference, distinguished from node-cache reuse.
6. Visual or listening assessment of the requested change and retained failures.
7. Hardware/resource qualification, Auto eligibility, model access/licensing,
   and publication approval, each recorded independently.

The family work includes scoped native and model-execution evidence, but it does
not establish universal quality, arbitrary composition support, cross-platform
resource safety, or a release-wide pass. FLUX strict detail/seam/reference quality,
full-weight native optional LoRA-scale demonstration, and exact resource recipes
remain separate acceptance work. Audio listening and publication decisions are
also independent of technical delivery. Historical migration and Gallery evidence
must satisfy their own current validators; do not infer approval from these docs.

Preserve creator settings during tests. Use explicit instance changes for resource
comparisons and record requested versus effective values. Keep dated run logs,
task IDs, media, hardware inventories, and handoff notes in private review storage;
share sanitized evidence through the relevant issue or pull request. Retained
results keep their original source identity and are not relabelled as a fresh run.
Follow the [engineering procedure](cluster-engineering-lessons.md) and the
[contributor validation commands](../CONTRIBUTING.md#validation).
