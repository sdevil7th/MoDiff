# Transformers depth workflows

In New → Depth estimation, select an installed depth model
and preview the connected workflow. Load Models and Predict Map use the same
ordinary graph, image input, output preview and Saved Block controls as other
workflows. The underlying task nodes are Load Depth Estimation Model and Predict
Depth; the two reviewed Depth Anything V2 variants share these nodes.

The loader uses the official Hugging Face Transformers AutoImageProcessor and
AutoModelForDepthEstimation APIs. It requires the reviewed Transformers base
runtime installed by ordinary setup, an immutable Hub commit or an explicitly selected local
model, local files, native classes and safetensors. Browsing or opening a workflow
does not install a runtime, download weights or authorize repository Python.
The execution profiles use the same required base-runtime contract as the existing
Transformers text and vision nodes. Heavyweight imports remain lazy. Models remain
in the existing graph cache.

## Inputs and outputs

Predict Depth takes one image. Processing Resolution `0` preserves the selected
processor's configured resize target; a positive value overrides that target.
The reviewed DPT processor preserves its configured aspect-ratio and patch-size
rules. The backend checks projected geometry before preprocessing, then verifies
actual tensor dimensions before model execution. Images with excessive aspect
ratios can require a connected Resize node. Match Input Resolution uses the
processor's official depth postprocessing to return the source dimensions.

- **Prediction Map** follows MoDiff's existing float32 NHWC relative-depth
  contract, with near `0` and far `1`.
- **Depth Preview** is an RGB grayscale visualization of that normalized map.
- **Native Depth** retains the model's floating-point values after optional
  spatial interpolation, as a CPU float32 HW tensor. It is not min/max normalized.
- **Depth Details** records the immutable model receipt, processed/output shapes,
  native range, depth polarity and normalization semantics. A constant map is
  explicitly identified.

Normalized preview intensity is not distance in metres. Relative inverse-depth
and model-declared metric depth use different polarities. The reviewed Depth
Anything configuration supplies the default; other models must specify their
documented polarity explicitly. A declaration of metric depth does not establish
measurement accuracy. Negative values introduced by upstream bicubic spatial
interpolation are retained in the native output rather than silently clipped.

## Reviewed adapter scope

The nodes are task-generic. The initial bounded preprocessing adapter uses the
native DPTImageProcessor contract; another AutoImageProcessor needs a reviewed
geometry/postprocessing adapter before execution. Selecting an arbitrary AutoModel
class alone cannot establish processor geometry, depth polarity or output units.
The two curated profiles pin the relative Small and Metric Outdoor Small models.
Auto resource qualification and Gallery eligibility remain separate from this
integration and require their own execution evidence.

The source contracts are the official
[Transformers depth documentation](https://huggingface.co/docs/transformers/model_doc/depth_anything)
and the reviewed runtime's `models/dpt/image_processing_dpt.py`,
`models/depth_anything/modeling_depth_anything.py` and
`models/auto/modeling_auto.py`. The executable package provenance and versions
remain in `modiff/optional_runtimes.py`; no additional model library is introduced.

## Optional prompt enhancement candidate

Model Manager uses the existing Install/Repair path for the pinned PUP revision.
Its selected closure contains the Safetensors model and eight processor,
tokenizer, chat-template, configuration and system-prompt files. This acquisition
inventory does not make the candidate public or qualify its resource usage.

Explicit Load repeats exact file-size and managed snapshot/blob containment
checks, then hashes only the eight bounded metadata files before either official
Transformers factory runs. Both factories receive that same checked snapshot
path; the model receipt retains the immutable Hub repository and revision.
The large model is not implicitly rehashed: its catalog SHA is publisher LFS
provenance, not a claim that local model bytes have been verified. A missing,
wrong-size or altered metadata asset requires Install/Repair before Load.


The existing image/text loader and generation node also support an explicit
`enhance_prompt` operation for one reviewed immutable model/processor pair.
`standard` remains the default and preserves image/video caption behavior.
The selected adapter uses the official Transformers factories, local files,
safetensors and no remote code. Its backend descriptor records the model,
processor, system-prompt and chat-template identities. This candidate adds no
public model profile, Auto eligibility, template permission or live qualification.

Enhancement accepts text alone or up to ten ordered images. It returns a prompt
proposal and keeps the authored request separately; it never rewrites a saved
prompt. Preview and explicit Apply remain the caller's responsibility. Semantic
fidelity is not established by a successful generation. The separate Enhancement
Token Limit permits at most 3,584 generated tokens; ordinary Max New Tokens stays
bounded at 2,048. The selected recipe uses greedy decoding with thinking disabled
and decodes only the generated suffix after verifying the input-token prefix.

The four pinned system prompts cover text, reference editing, and their target
layout variants. Source regions are refused without actual bound reference
identity. Generated structured layouts are diagnostic and cannot be applied
until a reviewed output parser validates their boxes and IDs against the authored
layout. The diagnostic result retains the original authored layout.

Enhancement-only controls are hidden on ordinary caption nodes and omitted from
historical registered Block roles and operation projections. An explicit backend
recipe overlay can expose them; visibility never authorizes an unreviewed model.
The exact current task is checked before normalization and rendering, during
bounded fingerprint reads, and around generation. An unrelated task or a stale
node interrupt flag is not cancellation authority.

Consumed-input evidence keeps the authored prompt separate from the actual
rendered template. It records the rendered UTF-8 and input-token fingerprints,
ordered normalized RGB fingerprints (including geometry), and actual processed
pixel/grid dtype, shape and content fingerprints before device transfer. A bounded
result projection retains the rendered text, reference dimensions and image grid;
raw tensors never enter history. These RGB fingerprints are observations, not
VAE-issued reference identities. Strict client receipt-parser pairing is pending.

A target-only structured result can now carry a separate `structuredProposal`
with a validated caption, proposed descriptions and the original target IDs and
boxes. The parser accepts one bounded trailing table in the selected task grammar;
it rejects duplicate/unknown IDs, unmatched tags, malformed or excessive JSON,
and any changed box. Authored coordinates must map exactly to the publisher's
integer grid; they are never rounded. Source rows and unsupported first-pass
estimate fields remain refused. Reference mentions in a caption address only the
actual supplied image order and do not authorize source geometry.

The raw generated result stays diagnostic. Even for a validated target proposal,
the ordinary text output remains `None` until a paired UI can apply caption and
layout together with Undo. `applicationAllowed` describes the validated preview;
`applyRequired` stays true and semantic fidelity remains unverified. The authored
prompt/layout are retained separately. This parser neither applies a saved edit
nor feeds a caption/JSON suffix automatically into Encode Prompt.

Image/text loaders register their actual neural module with the existing memory
manager. Standard captioning and enhancement lease that exact model at the normal
neural call boundary. Direct lower-level unregistered handles remain supported;
a manager-backed handle whose exact record was evicted must be reloaded. Owner
release preserves shared IDs and detached data, then refuses success if an
unshared registered neural object remains externally referenced. This proves
ownership cleanup, not physical memory reclamation or an execution schedule.

This first backend slice does not implement shared-stage UI Apply/Undo/persistence,
owner release before loading the image model, offload, quantization, or combined
memory qualification. PUP and the image model can overlap in the ordinary graph;
do not infer sequential release from a text connection or this helper. Existing
cache ownership and resource-lifetime contracts must prove that separately.

A genuine bound enhancement task also emits a hidden `proposal_summary` string.
Its closed, UTF-8 JSON is limited to 2 MiB and binds the original task/attempt,
node, worker/source/graph, exact parsed adapter declaration, authored-input
hashes, actual greedy controls, native model receipt and consumed-input evidence.
Browser correlation may be null. The current recipe is always null, so this is
only a diagnostic preview and cannot authorize Apply or a later task. Warm cache
hits retain the original producer and identical bytes. Standard and unbound
calls return `None`; their existing text/result values are unchanged.

The string can be served by the ordinary static cache route after owner release
only when the exact preloaded native class and its registered string output pass
preflight. The detached holder keeps verified module/action strings, no neural
state, and is reconstructed before later execution. Object outputs remain
unservable. Custom fields with the same name keep historical material-retention
behavior and gain no native identity. Historical role schemas omit the new output
unless that exact role declares it as an outgoing connection. Client parsing,
terminal/current-input joins, explicit ancillary execution scope and scheduling
remain separate prerequisites; this transport grants none of them.

When connecting the preview to a control model, use that model's documented
intensity convention. A compatible Image socket alone does not establish depth
polarity; an ordinary Invert processor can reverse the normalized preview.
