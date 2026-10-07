# Image template migration and validation

Image templates use the same ordinary operation-authoring transaction as the
developer workflow. Choose the template's exact model, task and execution
profile; do not substitute a family alias or a different task to obtain a native
graph. Where the reviewed implementation exposes encoding stages, the graph
contains real **Encode Inputs** nodes. A **Guidance** node is appropriate only
when a compatible guider is actually consumed by that route.

Whole-pipeline task implementations remain explicit. Retaining one is preferable
to presenting decorative stages that do not participate in execution. A missing
application integration and an unavailable upstream task are different reasons;
record the actual reason and revisit it when the reviewed implementation changes.
See [workbench acceptance](workbench-acceptance.md) for the broader release gate.

## Keep the validation claims separate

Migration, output comparison and platform qualification answer different
questions. Record each independently:

| Claim | Required evidence |
| --- | --- |
| A fresh template uses the current nodes | Exported visible graph and API graph, exact operation selection and application source identity |
| Its output matches the previous recipe | Completed old/new tasks, consumed recipes and runtime identities, complete ordered raw outputs and passing image comparison |
| A previously broken recipe now runs | The original failure plus successful current tasks; without an old image this is functional repair, not old/new image parity |
| Auto works in a memory mode | Cold and resident runs on the actual hardware, unchanged recipe, plans, reuse/reclaim observations and memory captures |
| Windows is qualified | Actual Windows installation and execution receipts; Linux and mocked platform tests are insufficient |
| A model is ready for public admission | The route's complete runtime, access, lifecycle, resource and output requirements in addition to graph construction |

A later source change does not erase earlier results or reassign their source
identity. Retain the original receipt and add the new checks with their own
scope. State which routes need another model run; do not describe completed
comparisons as missing simply because a broader qualification gate remains open.

Keep an evidence index outside Git with paths and hashes for source inventories,
task receipts, input/output sets, comparisons, visual notes and test logs. Retain
failed and incomplete attempts alongside later passing runs. An offline image
viewer helps review, but the indexed raw contracts and receipts establish what
ran. Private evidence and local Gallery caches do not arrive through `git pull`.

For the next design steps, see the
[Mellon comparison and simpler image workflows](simpler-image-workflows-mellon-comparison.md).
For routes beyond the image templates, see
[large image model validation](large-image-model-validation.md).

## Preserve the recipe before changing the graph

Retain the old creator contract and a fresh ordinary backend run. Preserve the
prompt and negative prompt, resolved seed, generator device and initialization,
dimensions, batch size, inference steps, guidance, scheduler configuration,
maximum sequence length, dtype, quantization and offload policy. For image tasks,
also preserve the exact input bytes, ordered references, masks and preprocessing.
Outpaint padding and stitching are part of the recipe, not cosmetic layout.

Guidance scales alone are insufficient. In the reviewed whole Z-Image pipeline,
scale 1 enables conditional/unconditional guidance with the original formulation.
The ordinary native developer starter disables it. The seven converted Z-Image
templates explicitly preserve the whole-pipeline behavior. Their app-owned native
blocks retain its combined model batch and FP32 time/prediction arithmetic; other
guidance techniques keep their upstream path. Check the actual guider receipt as
well as the scale when comparing these recipes. Encode Inputs retains authored
negative text when that guider requires unconditional conditioning; a disabled
developer starter continues to skip negative encoding. Check the actual encoded
text so an omitted field cannot silently substitute an empty negative prompt.

For FLUX.1 Dev/Krea, preserve the order of the latent-dtype cast and scheduler
time normalization. Their native stages use the whole pipeline's order through
a small app-owned denoiser adapter; scheduler updates and other model arguments
remain upstream. A real small-model counterexample establishes the rounding
difference, but full-model output parity still requires matched runs.

Qwen Image text-to-image also needs the whole pipeline's attention-mask
convention. After real native text-input validation, batch expansion and RoPE
preparation, its app-owned adapter removes an attention mask only when every
token is valid.
Padding masks and encoded tensors remain unchanged. An all-ones mask can select
a different SDPA execution path from no mask, even with identical embeddings;
compare actual model predictions and full outputs rather than assuming the two
forms are numerically interchangeable. This adapter is limited to the reviewed
text-to-image input branch; edit, layered, inpaint and ControlNet branches retain
their existing native behavior and require their own paired validation.

### Qwen Edit masked native templates

The pinned `QwenImageEditModularPipeline` has a genuine
`image_conditioned_inpainting` workflow, published under the operation task
`inpaint` and the Studio mode `modular_inpainting`. Its ordinary native encoder
uses the VAE posterior mode. The existing whole EditInpaint recipe samples that
posterior and uses a different normalization operation order; the app also
composites the decoded result with the original soft mask. Selecting the native
route alone therefore does not preserve the whole template's pixels.

Matching hidden `ModelsLoader` and `ImageEncode.inpaint_compatibility = "whole_v1"`
selectors enable conditional native adapters for that exact Qwen Edit inpaint
branch. The default remains `"native"`. The selected owner applies the existing
whole-route ROCm vision-attention policy at load time, and keys text-encoder/VAE
component reuse by that compatibility policy. An encoder cannot select it on
another owner's VAE. The compatibility path retains the pinned whole preprocessing and
crop geometry, samples the actual source posterior with the owned generator,
and removes all-valid text masks after real native input/RoPE preparation, as
the pinned whole inpaint encoder does. Padded masks remain unchanged. It also
carries an immutable source/mask snapshot through backend-issued route state
for the existing final mask composite. A graph cannot author that snapshot or
replace its downstream route state. Ordinary Edit, Edit Plus, Layered and the
Qwen text-to-image mask convention remain separate.

Fresh object replacement, mask draft, aspect outpaint and outpaint draft
templates now select that exact native route through the ordinary graph builder
and existing resource profile. They preserve the original dimensions, strength,
seed and optional crop padding. Outpainting uses the existing `OutpaintCanvas`
with the original margins, overlap, feather and fill; its actual canvas feeds
both prompt and image encoding. The builder requires the reviewed
`image_conditioned_inpainting` starter and owner binding before applying the
paired compatibility selectors. Generic developer starters remain native, and
opening a saved workflow preserves its explicit recipe.

The complete original 8/16/12/20-step recipes have matched-output checks on local
ROCm in both Custom and ordinary Auto execution. Seed and prompt changes followed
by restoration also check genuine generation on the same cached owner; an
identical-input cached result is recorded separately. These checks do not qualify
other hardware or every resource mode. The original requested token limit of 512
is absent from native Edit encoding: at the reviewed pin, whole `encode_prompt`
declares that argument but does not read it. Its separate input validation rejects
values above 1024. Keep that requested-interface difference and the raw whole
`attention_kwargs={}` versus native `None` distinction in comparisons; numerical
parity does not establish identical argument interfaces. Tiny real-VAE, processor
and generator differentials remain evidence for their specific tested boundaries.

Keep the whole pipeline's unused masked-image posterior computation distinct.
It runs after initial noise and consumes additional randomness, while the native
loop uses the source latents and mask. Equal source posterior, packed initial
noise, scheduler/model boundaries and final pixels do not establish equal final
opaque-call RNG state or identical executed work.

At this Diffusers pin the six remaining Flux templates keep their published
whole operations. Fill and its outpaint variant need the masked conditioning
channel layout of `FluxFillPipeline`; Canny/Depth use the concatenated control
latents of `FluxControlPipeline`, which are not ControlNet residuals. Redux edit
and multi-reference use the actual prior followed by the base pipeline, including
ordered reference weights and their distinct distilled guidance values. The
reviewed `FluxModularPipeline` text/image workflows do not supply those specialty
contracts. Visible stage labels or an unrelated control route cannot substitute
for a reviewed, executable decomposition.

FLUX true CFG and transformer distilled guidance are distinct settings. The
old image wrapper's primary guidance field sets true CFG; its secondary control
sets distilled guidance. Converted Dev/Krea and Kontext templates retain both:
the real Guidance owner controls positive/negative encoding and the two model
evaluations per step, while Denoise retains the separate embedded scale. Check
both consumed values, including empty negative prompts. The old Gallery leaves
the secondary override disabled, so converted Dev/Krea and Kontext templates
preserve the whole pipeline's effective embedded scale of 3.5. Do not infer it
from the separate developer starter, which enables a recommended override.
Schnell retains disabled true CFG; its model does not consume an embedded
guidance tensor. An ordinary native starter keeps
true CFG disabled until explicitly enabled; do not change its existing embedded
guidance when adding the optional guidance owner.

LoRA comparisons require the same base-model revision and ordered adapter set:
exact weight bytes, adapter names, scales, replace/append behavior and scheduler
overrides. Upscaling requires its own immutable weights and output dimensions.
The native LoRA node supports an explicit adapter name and an ordered
**Previous LoRAs** input; see the [Modular node guide](../modules/ModularDiffusers/README.md#lora).

Record what the backend actually consumed. A template default, requested random
seed or model revision placeholder is insufficient. A seed number alone does
not prove identical random inputs across CPU and GPU generators. Record actual
generator and scheduler observations when available; label source inference
explicitly when the runtime receipt cannot establish them.

Save the served application source identity, upstream Diffusers commit, package
versions, platform, accelerator and driver beside the graph and task receipts.
Keep old and new application identities separate: the implementation is expected
to change, while the consumed recipe and model identities must remain comparable.
Use isolated storage and extension discovery; leave existing models, workflows,
history and the operator's accelerator environment intact.

For source-only CPU schema or preprocessing probes, explicitly disable
`ExtensionStore.load_enabled` **before importing `modules`**. Private `CONFIG`
paths and `MODIFF_MANAGED_ROOT` alone do not prevent discovery of `custom` in the
working directory. Preserve the operator's extension approval file bytes and
hash before validation, alongside the model and operator-state snapshot. If a
file was not captured beforehand, report that gap rather than claiming its
contents remained unchanged.

## Compare actual outputs

Retain the complete ordered raw output set outside Git. Compare fresh old and new
runs on the same machine and upstream runtime. Existing Gallery pictures can be
historical visual references, but do not prove a migrated graph generated them.
An execution-selection change invalidates their claim to current exact-template
compatibility without deleting their provenance.

The backend provides a local comparison tool:

```text
python scripts/compare_image_template_outputs.py --baseline old.png --candidate new.png --baseline-contract old.json --candidate-contract new.json --report comparison.json
```

For multiple outputs, list all image paths after each image argument in output
order. Each JSON contract must declare
`format: "modiff.image-template-comparison-contract.v1"`, `templateId`, `task`,
`models`, `settings`, `runtime`, `inputs` and `expectedOutputs`:

- `models` contains the actual model set with `role`, `repoId` and an immutable
  40-character `revision` for every entry.
- `settings` contains actually consumed values and a resolved nonnegative integer
  `seed`. Include generator, scheduler, adapters and preprocessing when relevant.
- `runtime` records `torch`, `diffusersCommit`, `transformers`, `peft`, `platform`
  and `device`. The Diffusers commit is an immutable 40-character revision.
- `inputs` is the ordered list of consumed `role` and `sha256` pairs, or `[]` for
  a text-only recipe. Include masks and references without reordering them.
- `expectedOutputs` lists the complete ordered `width` and `height` values.

The tool rejects mismatched contracts, counts or dimensions. It compares decoded
RGBA pixels rather than PNG metadata and defaults to exact pixel identity. A
nonzero `--max-pixel-error` is an explicit numerical tolerance, not aesthetic
approval. It reports error measurements and returns a failing exit status when
the declared comparison does not pass. Do not broaden tolerance simply to turn a
regression green. Explain any difference using observed execution behavior.

Inspect images for prompt adherence, composition, detail, artifacts and task
fidelity. Numerical similarity and attractive output answer different questions.
An improved-output claim needs retained comparison evidence and visual review;
it cannot be inferred from a graph, hash or successful queue task. This tool
neither qualifies Auto nor publishes Gallery assets.

## Exercise the complete user journey

Run the actual template from the Gallery using the visible graph and ordinary
Run action. Check the initial load, cold generation and subsequent runs with a
changed prompt and seed. An unchanged cached output does not prove another model
forward. Observe loading, encoding, denoising and output, and verify the consumed
settings rather than only the creator fields.

Check Auto before and after the model is resident, after refresh, and after a
model change or release. Use the complete graph, including configuration and
media utility nodes. Confirm progress remains visible through reconnects and
that expected refresh disconnects do not spam errors. Exercise cancellation and
retry without submitting a second task while the first task's state is unknown.
Save and reopen the graph, check editable stages and Undo/Redo, and verify LoRA,
control, edit, inpaint, outpaint and multi-reference branches independently.

Capture read-only OS memory observations alongside task timestamps using
`scripts/capture_runtime_memory.py`. See
[resource monitoring](troubleshooting.md#resource-monitoring-during-execution).
Sampled peaks are lower bounds. Shared-memory AMD observations do not establish
dedicated NVIDIA VRAM requirements or a Windows fit guarantee. Advertised total
machine capacity and runtime free-memory demand are separate planner inputs.

## Windows handoff

Push the validated backend and client changes together, then pull both on the
Windows machine. Use the matching branch and update any client CI backend SHA
only after the real backend commit exists. Copy no Linux virtual environment or
unreviewed private runtime overlay. Preserve the Windows machine's existing
model cache and user data.

From a clean backend checkout on NVIDIA Windows:

```text
uv sync --extra cuda
uv run --extra cuda python -m modiff.preflight --json --check-port 8088 --fail-on-error
uv run --extra cuda python main.py
```

For client development, run `npm ci`, then `npm run dev` in the sibling client
checkout. Alternatively use the backend's validated bundled client. See
[developer setup](developer-setup.md) for versions, direct `uv pip` commands and
other accelerator profiles.

Template defaults are fetched from the pinned public Gallery dataset, verified
against their declared hashes and uploaded through the normal template flow.
Internet access is required; local ignored Gallery binaries need not be copied
from the Linux checkout.

First run a small supported text-to-image template in eager/SDPA mode, then repeat
with changed prompts and seeds. Confirm Transformers and PEFT need no optional
installation or activation. Check the allocator configuration, repeated-run Auto,
refresh/reconnect, LoRA and the remaining task branches with the same preserved
recipes. Test optional compilation separately after an actual kernel probe.

Record the actual Windows OS, GPU, VRAM, RAM, driver, runtime, model revisions,
plans, consumed settings and raw outputs. A Linux GPU run, a mocked Windows test
or a successful compiler import does not close this platform gate. Keep explicit
per-route failures and capacity limits; do not change default recipes to hide them.
