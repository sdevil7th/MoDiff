# Cosmos supporting safety dependency

Cosmos diffusion executes through the official Hugging Face Diffusers library,
currently pinned to `diffusers==0.41.0` in `pyproject.toml` with distribution
hashes in `uv.lock`. The separately pinned source-catalog commit is provenance
for reviewed declarations; it is not the installed distribution identity.

Diffusers 0.41.0 explicitly imports `cosmos_guardrail.CosmosSafetyChecker` in its
[official Cosmos Omni pipeline](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py#L50).
`cosmos-guardrail` is a separately published supporting safety package, not a
Hugging Face library and not an automatically installed Diffusers dependency.
Its PyPI project metadata attributes the code to NVIDIA; the reviewed package
contract identifies the distribution publisher as cosmos-guardrail contributors.
Do not infer ownership from the model's Hub organization or a compatible API.

MoDiff explicitly accepts this narrow upstream-supported safety integration for
the reviewed Cosmos Omni and Distilled paths. It does not authorize another
diffusion runtime, graph executor, library-owned workflow, arbitrary model code,
or general third-party model-library support. The diffusion pipeline, schedulers,
components and Modular blocks remain Diffusers-owned. All safety resources run
locally under the same model owner and graph/resource/security boundaries.

The exact supporting distribution is
[`cosmos-guardrail==0.3.1`](https://pypi.org/project/cosmos-guardrail/0.3.1/).
Its reviewed universal wheel has SHA-256
`58c0ff49393f65b7589aa8365edc6640ebbcdfbfdde69c95b8a088fe13237ed6`.
The package closure, delivery platform, dependency versions, wheel URLs and
hashes are executable declarations in
[`cosmos-safety-runtime-artifacts.v1.json`](../data/cosmos-safety-runtime-artifacts.v1.json).
An explicit managed Install action and version/artifact verification remain
required. Registry discovery, browsing and planning never install the package.
See [optional runtime delivery and qualification](optional-runtime-optimizations.md)
for the separately qualified Linux delivery and platform limits.

The owned adapter uses the upstream safety algorithms with immutable local
artifacts: `nvidia/Cosmos-Guardrail1` at
`d6d4bfa899a71454a700907664f3e88f503950cf` and
`Qwen/Qwen3Guard-Gen-0.6B` at
`fada3b2f655b89601929198343c94cd2f64d93cc`. The Qwen text guard uses the official
Transformers loader with safetensors and no remote Python code. The supporting
RetinaFace detector performs face-blur postprocessing; it is not a general
image-content classifier. Exact files and hashes are checked by
[`cosmos_safety_contract.py`](../modiff/cosmos_safety_contract.py) before the
adapter attaches the checker. Required safety configuration remains enabled and
missing, incomplete or disabled checks fail closed. Access to gated artifacts is
a separate prerequisite.

## Fresh text-to-image stages

Fresh Cosmos Omni text-to-image operation/template graphs use Load Models,
Encode Inputs, Denoise, Decode Latents and Preview Image. TextEncode retains the
mandatory text check; Decode retains Diffusers' mandatory output safety
processing and the same owned
checker configuration. The official
[`Cosmos3ActionOutputStep`](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/modular_pipelines/cosmos/after_decode.py)
returns `action=None` for this image task, so its AfterDecode side-output stage is
omitted from fresh image operation/template declarations. The raw SDK source
catalog retains the authoritative upstream hierarchy, including this block.

The historical `WorkflowCosmos3OmniAfterDecode` node remains executable. Saved
graphs, immutable upstream workflow snapshots and historical output evidence
are preserved. The exact original managed image specification from backend
`661efe87a40c122b0692e32dde7e8bd55ed98ceb`, hash `studio-spec-v1-91ed105d`,
is archived in the
[compatibility contract](../data/studio-execution-spec-compatibility.v1.json)
and accepted only when a saved receipt explicitly selects that identity. New
authoring uses the current specification; unknown hashes remain rejected.
Every video/action declaration remains unchanged. Removing the
image-only no-op does not promote Auto, another artifact, another platform or
public Gallery eligibility.

No-download regression tests verify image-stage removal, historical node and
upstream-snapshot preservation, every other Cosmos adapter, mandatory safety
configuration, local artifact checks and owned resource attachment. These are
contract/CPU proofs. A fresh original-model cold run and changed-seed resident
run on the reviewed device are still required to establish live output and
lifecycle acceptance for the changed graph.


## Selected text-to-image component adapters

The native component adapter is limited to the current Super text-to-image
execution specification, `nvidia/Cosmos3-Super-Text2Image` at
`daf3d374804be4c512c2135568a7cb95d4341d79`, above the loader boundary. It uses the actual official `Cosmos3OmniTransformer` PEFT
lifecycle with the actual owning pipeline. It does not add pipeline LoRA APIs
or accept a context-free call to the ordinary adapter updater.

The reviewed route requires literal `offload_mode=none`, `auto_offload=false`,
CPU weight staging, no quantization or externally supplied components, and no
remote Python code. Other valid vanilla placements continue through their
existing loader path. An adapter request outside this route fails before an
existing owner is mutated or replaced. Only ordinary full-prefix paired
`transformer.*.lora_A.weight`/`lora_B.weight` factors with matching ranks and
actual target shapes are accepted; scheduler replacements, DoRA, foreign
prefixes and unrecognized variants are refused.

The selected source task, executable stage/block selectors, one-frame input,
immutable selected artifact, reviewed local index and backend source identity
bind the adapter context. Canonical saved integer strings are normalized by
the existing declared numeric input rules. Nano's source graph profile and a
reviewed Super artifact variant remain separate identities. A numerical frame
value alone cannot authorize an image adapter.

The loader binds successful completion to its actual pipeline, component,
mandatory checker, output, method implementation and ordered immutable adapter
descriptors. Changed or incomplete state is rebuilt through existing scoped
owner release; shared or externally retained target/checker references stop
replacement rather than being forcibly cleared. Failed injection,
activation, cancellation or checker publication remains incomplete until a
fresh successful owner is constructed. Wrapper/reference disappearance is
not a physical RAM reclamation or model-fit proof.

The selected Super authoring policy requires the current native model owner
to retain literal NONE placement, Auto Offload disabled, CPU weight staging,
remote code disabled, and optional quantization/external-component inputs at
their declared unbound `null` values. The selected task and immutable pair
must still match the current source graph. Missing or ambiguous controls do
not authorize adapter editing, and a historical no-adapter receipt does not
grant adapters. Auto must preserve these constraints using an already
admitted recipe; this policy does not make the Super route Auto-qualified.

Authoring permission and any declared required owner controls remain separate
from readiness, Auto eligibility, artifact access, model fit and live
qualification. Required safety remains enabled and uses the same checker,
ownership and execution contracts described above. This integration does not
install or activate the supporting safety runtime. In particular, Windows
delivery remains `candidate_unqualified` with Install and Activate unavailable
until its separately reviewed delivery is qualified. Tiny synthetic component
and ownership tests do not qualify trained adapters, generated outputs or
safety algorithms; original trained-model cold/repeat/changed-adapter and
mandatory-safety output evidence remains required on the admitted platform.
