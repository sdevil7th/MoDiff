# Larger image models: integration and validation

The image-template migration is a bounded set of recipes, not the full Diffusers
catalog. A model can have application integration and an exported graph while
its actual execution, memory policy or public admission remains unqualified.
Use the backend execution specifications and artifact reviews as the authority
for an exact model, task and profile. Do not infer readiness from a palette entry
or `native_integrated` research disposition.

## Candidates beyond the image templates

| Model or family | Existing integration | Work remaining before a public working-model claim |
| --- | --- | --- |
| `black-forest-labs/FLUX.2-dev` | Whole-pipeline and native modular paths for text-to-image and reference editing | Reviewed access/license and runtime, full unmodified recipes, native/whole output comparison, lifecycle and actual device-memory checks; public execution admission remains closed |
| `zai-org/GLM-Image` | Ordinary whole-pipeline text-to-image path | Live execution and output/resource/lifecycle evidence for the exact expert route; editing is outside its initial admission |
| `Qwen/Qwen-Image-2.1` | Ordinary text-to-image, edit and ordered-reference paths; reviewed revision/runtime and bounded full-weight Linux text/two-reference edit successes already exist | Broader release/reference-replacement and dedicated-device/resource coverage; preserve the existing [Qwen 2.1 evidence](qwen-image-21.md#runtime-and-current-qualification) rather than calling the model entirely untested |
| `jdopensource/JoyAI-Image-Edit-Diffusers` and `jdopensource/JoyAI-Image-Edit-Plus-Diffusers` | Ordinary image-editing paths | Resolve artifact/license payload requirements, then validate actual inputs, outputs, runtime and resources |
| `black-forest-labs/FLUX.2-klein-9b-kv` | Whole-pipeline KV editing and a separate native text-to-image alternative | Actual KV-edit execution and reference/cache behavior; the native alternative does not prove native KV-edit support |
| `nvidia/Cosmos3-Nano` and `nvidia/Cosmos3-Super-Text2Image-4Step` | Reviewed static modular graph construction | Required guardrail/access/runtime integration and actual device execution before enabling Run, Auto, templates or Gallery; the distilled text-to-image route uses its distinct reviewed class |

These are not documented failures caused solely by insufficient VRAM. More GPU
memory can make the execution checks practical, but does not resolve a missing
runtime component, an unopened admission gate or a model-access prerequisite.
Publisher estimates and NVIDIA examples do not establish AMD performance or fit.

Other research blockers are similarly specific. HiDream-I1 needs its reviewed
external text-encoder/composite-loader requirements; HunyuanImage 2.1 has an
artifact/license contract boundary; Nucleus Image and the reviewed SD3 Medium
artifact also have disk-capacity blockers recorded for the reviewed machine.
Disk shortfalls are not VRAM
measurements. The SD3 Medium artifact's status does not apply to every SD3/3.5
route. Klein Base 4B already has separate live execution evidence, so it should
not be described as an entirely untested model.

The route definitions live in
[`modiff/studio_execution_specs.py`](../modiff/studio_execution_specs.py).
The research inventory is
[`data/image-prototyping-readiness.v1.json`](../data/image-prototyping-readiness.v1.json).
The specific blockers are recorded in the corresponding `data/*-artifact-review.json`
files. Inventory counts and historical research defect labels are not a count of
currently failing templates.

## Use a large AMD GPU effectively

Prepare and push the paired backend/client commits before provisioning, following
[AMD cloud preparation](amd-cloud-qualification.md). Inspect the actual assigned
GPU architecture, VRAM, host RAM, storage and runtime after provisioning. Similar
product names are insufficient: an MI100 and an MI300X do not use the same
reviewed profile. Memory advertised for a board need not equal memory assigned
to a VM. Multiple devices do not automatically form one shared VRAM budget.

Start with a known small visible-frontend generation and a representative large
existing recipe. Exercise cold loading, a changed prompt and seed with resident
weights, cancellation/retry, refresh/reconnect, model release and another model
selection. This checks the dedicated-GPU behavior left open by shared-memory
Linux testing and addresses the original second-generation Auto complaint.

Then prioritize FLUX.2-dev when its exact access/runtime gate is ready: preserve
the full recommended resolution and step count, compare reviewed native and
whole recipes, and check text-only, one-reference and ordered multi-reference
execution. Weight size on disk is not peak device-memory demand. Offload and
quantization are separately labelled recipes, not silent changes to a failed
baseline. Keep text encoding local through the application's existing runtime.

Use GLM-Image, Qwen 2.1, JoyAI and Klein KV as subsequent candidates according to
their remaining prerequisites. Attempt Cosmos Nano before the larger distilled
model only after the mandatory guardrail and ROCm runtime path is reviewed.
Additional VRAM cannot substitute for those checks.

Run memory-intensive jobs sequentially on one GPU so task attribution, residency
and measurements stay interpretable. Windows installation and UX checks on a
different machine, source review, CPU contracts and result inspection can run in
parallel. A ten-hour GPU reservation is a time budget for a prioritized subset,
not a promise to qualify every remaining route.

## Evidence required for each tested route

Retain the exact paired application commits and served source hashes, model and
auxiliary revisions, upstream library versions, actual hardware/runtime, visible
and API graph exports, ordered input hashes, consumed recipe, task IDs, raw
outputs, plans, progress/cancellation outcomes and timestamped memory captures.
Use the [image comparison procedure](image-template-validation.md) for matched
outputs. Record sampled memory peaks as lower bounds and preserve failures.

A successful single generation closes that execution attempt. Repeated-run Auto,
different memory modes, image quality, Windows compatibility and public admission
require their own evidence. Do not rewrite prior proof files or enable route
admission merely because cloud preparation or one task passed.

## Upstream references

- [FLUX.2-dev model card](https://huggingface.co/black-forest-labs/FLUX.2-dev)
- [FLUX.2 pipeline documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux2)
- [GLM-Image model card](https://huggingface.co/zai-org/GLM-Image)
- [Qwen-Image 2.1 model card](https://huggingface.co/Qwen/Qwen-Image-2.1)
- [JoyAI Image Edit pipeline](https://huggingface.co/docs/diffusers/api/pipelines/joyimage_edit)
- [FLUX.2 Klein KV model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv)
- [Cosmos3 Super 4-Step model card](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image-4Step)
- [AMD Instinct product information](https://www.amd.com/en/products/accelerators/instinct/mi300.html)
