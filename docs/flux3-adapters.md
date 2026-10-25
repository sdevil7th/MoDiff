# FLUX 3 adapter execution boundary

The selected native FLUX 3 contract uses the ordinary Modular Diffusers loader,
generic LoRA descriptor chain, NodeBase dispatcher and component manager.
It requires a reviewed runtime containing the official FLUX 3 APIs. A source
audit does not replace the installed dependency contract or publish a template.

Normal workflow authoring exposes Klein and Klein Base through the explicitly
verified evaluation runtime and **Custom** memory. Each exact text-to-image or
image-edit profile declares its existing append-v3 prompt-enhancement parent and
an exact `rawAuthoringRecipeId` alternative. The ordinary starter request selects
the append declaration; **Enable enhancement** remains a separate user action.
That action selects plain wording for a null layout and append wording for an
authored layout, after checking that both declared parents have the same graph
and unchanged base specification. This attaches an editing offer without
creating a PUP branch or changing provider precision. Older plain, append and
strict-layout recipes keep their existing identities and hashes.

Raw edit offers use the existing plain `prompt-preview-authoring:v1` parents.
Raw text-to-image offers use fixed FP32/`cuda:0`/no-quantization/4096-input-token
parents: the existing Distilled `prompt-preview-fp32-authoring:v1` and the
additive Base identity `flux3-klein-base:text-to-image:prompt-preview-fp32-authoring:v1`.
The new identity leaves all fourteen earlier full declarations unchanged. A raw
Preview still refuses an authored layout; an append Preview still requires its
validated authored regions. Metadata alternatives do not grant Auto, hardware
fit, native model acceptance or public template eligibility.

Text-to-image offers additionally declare bounded `rawAuthoringRecipeChoices`
with exact recipe IDs and backend-owned labels. The default remains FP32; the
operator can explicitly choose the existing raw BF16/NF4 `prompt-preview-authoring:v1`
parent before Enable. This is an authored recipe choice, never an automatic
precision or format fallback. Explicit raw choices refuse nonnull layouts;
authored-region append remains the separate unchanged FP32 declaration. The
existing Preview input checks retain each selected recipe's declared literals.
NF4 still requires its separately installed/activated reviewed runtime, NVIDIA
CUDA compatibility and actual resource fit. No NF4 runtime or model-output
qualification follows from publishing this authoring metadata.

An explicit dense Image/Text load checks a necessary parameter-storage floor
before either processor or neural-model factory. For an already verified
single-file snapshot, the bounded Safetensors header counts floating weights
at the selected dtype and retains integer/bool buffers. If CPU loading or the
selected device cannot physically hold even those parameters, loading refuses
with the required and available total bytes. It never silently chooses NF4,
another dtype or another device. Shared-memory accelerators use physical system
RAM; device recommendations and planning caps are not hard capacity proofs.
Passing this floor is not a free-memory, activation,
peak or fit guarantee; unsupported metadata or unknown capacity supplies no
estimate. Quantized loads keep their separate reviewed storage policy. On a
constrained Windows GPU, choose the explicit BF16/NF4 raw recipe before Enable
and inspect its runtime readiness rather than assuming the default FP32 fits.

**Add LoRA** uses the existing generic ordered descriptor chain for those two
immutable model revisions. Its graph-edit policy requires NONE offload, Auto
Offload disabled, CPU weight loading, no component/quantizer overrides and no
remote code. The source-selected transformer lifecycle still validates the
actual graph, adapter artifact, storage format, placement and runtime; authoring
permission is neither resource fit nor model qualification. Other placements
remain inspectable/executable only under their existing separate contracts.
The pinned FLUX 3 workflows declare text-to-image and image-conditioned editing,
not a ControlNet workflow or reviewed ControlNet component. They therefore do
not offer a working ControlNet addition; reference-image editing is not ControlNet.
These private authoring controls do not publish templates or expand Auto.

The current private execution policy admits **unmerged transformer LoRA**
for the exact Klein and Klein Base immutable artifacts and their declared
`text2image` and `image_conditioned` workflows. The actual current native stage
wiring determines task authority. Labels, saved authoring hints and caller-made
identity tuples cannot grant it. The encoder and VAE are not adapter targets.

For `image_conditioned`, the reference image may pass from ordinary `Load.image`
through a finite chain of built-in `ProcessImage` Resize outputs before
`ImageEncode.image`. Each processor must select literal
`BuiltinImageOperationV1` / `image_upscale`, with no connected scalar or mask
suppliers. Custom/model image producers and other processing operations require
separate review. All media nodes must be executable and outside loops. This
media boundary does not replace the exact model-stage, owner, artifact or
placement checks. Processor topology participates in route identity; image
bytes and resize dimensions remain sealed by the existing reference binding
and input-cache invalidation, so a resize edit need not replace model weights.

Use Safetensors containing ordinary `transformer.*.lora_A.weight` / `lora_B.weight`
pairs, or the corresponding `diffusion_model.*` names handled by the official
Flux3 converter. Mixed prefixes, text-encoder keys, DoRA, unmatched ranks and
unknown target shapes are refused. The selected policy checks finite factor
values in bounded CPU slices, permits ranks up to 1024, and limits each adapter
file to 2 GiB. Immutable descriptor hashes and local/cache boundaries remain
mandatory. Adapter scales retain the existing generic descriptor limits.
Embedded `lora_adapter_metadata` requires the closed reviewed PEFT configuration:
transformer-only target names, matching ranks, finite alpha/dropout and supported
ordinary inference settings. Known disabled/null fields retain exact types;
active DoRA, RSLora, QALora, bias, saved modules and patterned overrides remain
refused. Duplicate/unknown fields are refused. The original metadata is passed
unchanged to the official loader; MoDiff does not implement PEFT scaling.
Inert Safetensors metadata remains permitted.

This policy requires CPU weight loading, no distributed placement, and no
component or additional quantizer overrides. NONE placement requires Auto
Offload disabled. The private source-selected context also declares Model CPU,
synchronous Group CPU and Group Disk with literal Auto Offload enabled; a hook
attribute or caller-made tuple cannot select those contracts.
Native storage retains its dense-only checks. The declared BFL FP8r and NVFP4
single-file formats additionally require the exact immutable weight/config
receipt joined to the actual loader, ComponentSpec load ID, component manager,
official quantizer and layer scheme. The ordinary single-file loader records a
single-root CPU load map; actual tensors must reside on the selected execution
device after NONE placement. This load metadata does not grant distributed
placement. Offloaded targets require the additional typed placement contract.
Logical adapter dimensions come from official layer input/output features,
not the packed NVFP4 weight shape. Other model and adapter policies retain their
existing behavior.

Model CPU preserves the ordinary manager's genuine `CustomOffloadHook` during
injection. Its exact model/manager collection, peer hooks, selected device and
strategy are sealed; another neural collection prevents manager reconfiguration.
Group CPU and Group Disk inject on a fresh CPU target before installing official
hooks. Group Disk uses the configured absolute plain temporary root and a fresh
per-load lease. Its official file paths, inverse tensor maps, complete parameter
inventory and file identities/full SHA-256 are sealed and checked before warm
reuse. Actual task-bound file verification remains cancellable and reports byte
progress. Invalid storage never falls back to the working directory. Existing
unselected offload callers retain their prior defaults and storage location.

Named adapters preserve chain order and individual scales. Zero scale disables
their contribution. A single ordinary `Lora` node supplies one descriptor; an
ordered chain supplies a descriptor list. Both use the same source-selected
artifact, component and placement checks. Removing or replacing a descriptor
invalidates the owning runtime and dependent cached material. The existing
executor releases the exact
incompatible owner before constructing its replacement. Surviving shared or
external target references stop replacement. A partial official load or
activation cannot publish a reusable complete owner; retry requires a clean
target. These are object ownership checks, not physical RAM reclamation claims.

Fusion/unfusion, training, full quantized model saving, compiled hotswap and
other offload combinations have no grant from this contract. Tiny CPU BFL tests
exercise reference dequantization rather than accelerated CUDA kernels. No
public adapter-authoring, Auto, template, trained model, hardware-fit or
image-quality qualification is implied.

For the exact reproduced Windows Torch `2.11.0+cu128` / CUDA 12.8 build
`70d99e998b4955e0049d13a98d77ae1b14db1f45` with capability 8.9 or newer, and
Linux Torch `2.9.1+rocm7.2.0.git7e1940d4` / HIP `7.2.26015-fc0010cf6a` build
`7e1940d4b11fd6128be4c42ba41567ec5ab87102` on `gfx1151` / capability `(11, 5)`,
MoDiff selects the documented official BFL per-layer fallback. The execution
mode is `fp8-storage/per-layer-dequantized-linear`; its actual compute dtype is
recorded separately. The original FP8 parameter, FP32 row scales, official
quantizer and exact layer class remain intact. Each forward uses Diffusers'
`dequantize_fp8r` and ordinary `torch.nn.functional.linear`, before the temporary
weight is released. Activations are not FP8-quantized on this path. This follows
the [official BFL compatibility behavior](https://huggingface.co/docs/diffusers/main/en/quantization/bfl),
not a conversion to another checkpoint or a native FP8 GEMM claim.

The adapter is installed only on a freshly loaded exact owned transformer,
before offload and PEFT hooks. Existing forward/hook mutations, another owner
and incompatible schemes are refused. The mode participates in loader and
component reuse identity; artifact, source, LoRA and placement checks remain
required. Unknown runtime rows retain their existing upstream execution and
do not borrow this compatibility selection. Native and NVFP4 storage are
unchanged. The transient FP32 dequantization and compute-dtype weight can add
peak memory: this does not establish faster execution, half peak memory, Auto
fit, public template eligibility or trained output on either host.

`tests/test_flux3_native_lora.py` exercises the ordinary source graph and loader
with genuine tiny official CPU components. Only component acquisition is
substituted. It deliberately stops before image stages and trained media;
native stage tensor composition is covered by `test_flux3_tiny_stage_dispatch.py`.
`test_flux3_bfl_lora.py` uses genuine local tiny single-file checkpoints, the
official ComponentSpec/quantizer/PEFT APIs and the same graph ownership lifecycle;
its synthetic artifact declarations do not stand in for pretrained files.
The placement tests exercise real official hooks, files, converter/PEFT and warm
owner lifecycle. Their explicitly limited CPU mechanics fixtures keep CPU
onload/offload and do not establish CUDA transfer, trained output or memory fit.
The opt-in tiny CUDA test is a separate root-run lane, not a default CPU gate.

## Optional no-weight processor test fixture

The existing private `:prompt-preview-authoring:v1` recipes retain their exact
plain-only declarations and hashes. The explicit Distilled and Base text-to-image
and edit-image `:prompt-preview-authoring:v2` recipes select the generic
`preserve-authored-v1` layout policy. They preserve every authored region ID,
source reference binding, source/target box and action, changing only caption and
description text. Native IDs map by exact authored ID or a unique exact
source-index/boxes/action signature; additions, omissions, ambiguity, unsupported
operator fields and non-grid coordinates refuse Apply. No geometry is rounded.
The pinned publisher system-prompt file stays unchanged. Plain requests and v1
declarations retain its exact prompts and hashes. For v2 authored layouts, a
separate authoritative system instruction replaces automatic ID/row generation:
the supplied table is complete, IDs and all geometry/source/action values stay
fixed, and no scene, anchoring or preserved-background rows may be added. Both
preallocation and generation fingerprint the actual composed system instruction;
v2 verification metadata declares those same per-mode hashes. Existing v2 drafts
must explicitly rebind to the updated declaration rather than silently adopting
a new hash. Structured image editing asks for the edit instruction itself and
the complete five-key table, with a single format example; unauthored scenery and
lighting remain untagged prose, while authored regions keep their IDs and rows.
The user request keeps the exact authored table last, in the mode's key order.
All static labels and formatting belong to the
hash-declared preservation request template (the legacy-named suffix field);
only authored text, allowed tags and table values are substituted. It grants no
reference authority. Version-two proposals retain both raw native caption/table and the
mapped caption so the client and bound summary independently check the mapping.
The declaration's `sourceRegions: false` continues to prohibit creating source
regions or issuing reference identities. `layoutPolicy` permits only retaining
already authored source DTOs; the ordinary downstream encoder still checks their
current reference binding before diffusion execution.
Apply remains explicit and atomic under the existing stale/source/scope guards.
Software fixtures do not establish successful trained structured output.

Rejected structured Previews expose a hidden `proposal_diagnostic` string for
inspection through `GET /cache/{node}/proposal_diagnostic?with_metadata=1`.
It contains the exact decoded suffix, without granting Apply authority. The
existing native observation binds it to its original task, output incarnation,
request and worker/source identity; the envelope includes both summary and
diagnostic UTF-8 hashes. Plain cache access, arbitrary object outputs and stale
or replaced observations remain refused. Public `text` stays empty, and neither
the strict parser nor the proposal summary schema changes. A diagnostic exceeding
262144 characters or 1 MiB of UTF-8, or containing invalid Unicode, is omitted
instead of truncated. Ordinary/plain and successfully validated proposals omit
this field's value. Normal Preview self-release retains the already bounded and
sealed diagnostic beside the summary. Later explicit cache release may remove
it; no historical output is minted or recovered by a read.

The separate private Distilled text-only recipe
`flux3-klein:text-to-image:prompt-preview-fp32-authoring:v1` fixes the existing
provider to FP32, no quantization, `cuda:0` and 4096 input tokens. It retains the
ordinary parent graph and Preview executor, with no reference-image route.
Select this exact current declaration through normal authoring/import; changing
the fixed NF4/BF16 literals in an older T2I recipe does not select it. Its provider
and request literals are enforced before Preview; all older v1/v2 declarations
remain unchanged. This private recipe is not Auto, hardware or native T2I proof.

The additive private
`flux3-klein:text-to-image:prompt-preview-fp32-authoring:v2` retains those exact
FP32/no-quantization provider literals and input limit, while selecting
`preserve-authored-v1` for structured T2I. It uses the same generic nodes, strict
parser and atomic Apply/Undo; target IDs, boxes and actions remain authored
authority. It has no reference-image route. All nine earlier identities keep
their declarations and hashes, including the plain-only FP32 v1 and existing
NF4/BF16 recipes. Metadata availability does not establish native structured
T2I output or another hardware/runtime qualification.

Reviewed prompt enhancement checks each generation score row through the official
Transformers logits-processor interface. NaN, positive infinity or a row with no
finite token candidate stops generation before token selection and publishes no
proposal. Negative-infinity token masks remain valid alongside finite candidates.
Scores and generated punctuation are never sanitized or filtered; the error asks
the operator to inspect the selected precision and runtime before retrying. This
numerical check does not diagnose an upstream kernel or qualify another precision.

For the exact declared PUP artifact, dense BF16 on the primary ROCm `gfx1151`
device uses the official model-local `set_attn_implementation` API to select
`vision_config=eager`; root and text attention remain SDPA. This narrowly scoped
compatibility policy follows an observed same-weight, same-input comparison:
default vision SDPA produced nonfinite scores, while vision-only eager produced
finite scores. It is not a kernel diagnosis or another device qualification.
FP32, NF4, CUDA and other architectures retain their existing defaults. No
weights, precision, forward functions or global attention backends are changed.
The policy is derived from the existing sealed source/revision, precision,
quantization and device fields plus the actual architecture; the public receipt
schema stays unchanged. Generation checks the live configuration of that owned
model. An older cached SDPA handle or changed attention state refuses with a
request to reload; use normal Release/Load before another Preview. A raw receipt
alone does not attest the actual attention configuration.

The ordinary backend gate requires no local PUP files. Tests that need its
reviewed processor metadata explicitly skip when `MODIFF_PUP_PROCESSOR_FIXTURE`
is unset. A supplied missing, incomplete or changed fixture fails. A skip is
not PUP execution coverage; run the fixture lane separately on each reviewed
runtime. `test_prompt_enhancement_install.py` substitutes a tiny weight file
for acquisition-boundary checks; it never validates the trained model.

To prepare the fixture, save the following as a temporary Python script outside
the checkout and run it from the backend root with a new absolute destination
argument. It downloads only the eight exact metadata files through Hugging Face
Hub, verifies their committed hashes, and constructs the actual local processor
without loading neural weights or permitting remote code. Existing Hub access
is required; the script does not accept model terms. Keep the destination outside
Git and do not publish its gated metadata or credentials.

```python
import hashlib
import json
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download
from transformers import AutoProcessor

review = json.loads(Path("tests/fixtures/pup-native-install.v1.json").read_bytes())
root = Path(sys.argv[1]).absolute()
root.mkdir(exist_ok=False)
processor = root / "processor"
processor.mkdir()
for row in review["files"]:
    if row["path"] == "model.safetensors":
        continue
    source = Path(hf_hub_download(review["repo"], row["path"], revision=review["revision"]))
    body = source.read_bytes()
    assert len(body) == row["byteSize"]
    assert hashlib.sha256(body).hexdigest() == row["sha256"]
    shutil.copyfile(source, processor / row["path"])
AutoProcessor.from_pretrained(processor, local_files_only=True, trust_remote_code=False)
receipt = {"status": "passed", "revision": review["revision"],
           "tokenizerSha256": hashlib.sha256((processor / "tokenizer.json").read_bytes()).hexdigest()}
(root / "result.json").write_text(json.dumps(receipt) + "\n", encoding="utf-8")
```

Set `MODIFF_PUP_PROCESSOR_FIXTURE` to that destination, then run the ordinary
backend gate to include these tests. For a focused check on Linux:

```bash
export MODIFF_PUP_PROCESSOR_FIXTURE=/absolute/path/to/reviewed-pup-processor
./scripts/with-runtime-env.sh .venv/bin/python -m pytest -q tests/test_prompt_enhancement_install.py
```

On PowerShell, use `$env:MODIFF_PUP_PROCESSOR_FIXTURE` and
`.venv/Scripts/python.exe`. The actual processor and synthetic neural tests in
`tests/test_prompt_enhancement.py` use the same fixture and remain separate from
trained PUP output, hardware capacity, Auto and browser qualification.


### Append details with an unchanged authored layout

The private Distilled edit-image `prompt-preview-authoring:v3` and explicit
FP32 text-to-image `prompt-preview-fp32-authoring:v3` select the additive
`append-details-preserve-layout-v1` policy. Their generic
`GenerateImageVideoText` operation is `enhance_prompt_append_details`; the hidden
operation is never substituted into an older recipe. Existing standard, plain
enhancement and strict `preserve-authored-v1` behavior remain unchanged.

The corresponding private Base identities are
`flux3-klein-base:edit-image:prompt-preview-authoring:v3` and
`flux3-klein-base:text-to-image:prompt-preview-fp32-authoring:v3`. They retain the
existing Base graph, Guider, execution profile and ordinary parent controls. The
edit recipe retains explicit provider precision choices; the text-only recipe
fixes FP32, `cuda:0`, no quantization and 4096 input tokens. All twelve earlier
declarations and their hashes remain unchanged. Both use the same generic
wording call, composition, strict summary verification and atomic Apply/Undo;
they grant no additional Auto/NF4 permission, public visibility, hardware fit or
trained Base-output qualification.

This policy makes one ordinary local PUP call with the immutable publisher's
plain `t2i` or `i2i` system instructions. Its model request contains the authored
caption with validated region/reference tag delimiters removed, followed by
bounded region ID/description wording context. It contains no layout geometry,
actions, source bindings or table authority. The exact rendered request, actual
operation, system hash, references, processor inputs, precision and task/source
observations retain the existing preallocation and native cache seals. Auto/NF4
request qualification is unchanged; these private recipes do not expand Auto.

The actual decoded native text is retained without trimming. It must be bounded,
nonempty valid Unicode and contain no angle tags, Markdown fences or genuine
trailing JSON table. Ordinary punctuation, quoted wording, styles and math are
not silently sanitized or rejected. The composed caption is exactly the original
authored UTF-8 caption, two newlines and the accepted native text. The entire raw
authored layout is deep-copied unchanged, including descriptions, row order,
schema, finite non-grid coordinates, IDs, source bindings and removal actions.
The authored tags are independently validated before model work: every region
ID appears exactly once, unknown/incomplete/nested tags are refused, and reference
tags must identify actual ordered images.

The opt-in `structuredProposal` has schema version 3 and kind
`caption-append-preserve-layout-v1`, with only `caption`, `layoutConditioning` and
`nativeText` beside the schema/kind fields. It has no fabricated `rawCaption` or
`rawTable`. The backend independently recomposes it from the original authored
request and actual native text before sealing the summary. The client separately
checks the selected current recipe, exact prefix, exact layout and new operation
before one atomic, undoable Apply. `layoutOutputValidated: true` means the local
composition preserved the authored layout; it is not a claim that the model
produced or validated geometry. Semantic fidelity remains unverified and needs
operator review. Rejected native details remain a sealed diagnostic, with no
automatic text feed or Apply. Both the caption and complete proposal retain their
existing finite Unicode, UTF-8 and JSON budgets. No native-model acceptance is
claimed by this source change.
