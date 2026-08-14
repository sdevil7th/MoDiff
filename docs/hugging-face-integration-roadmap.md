# Hugging Face Integration Roadmap

This document is the implementation and completion tracker for closing MoDiff's
official Diffusers, Modular Diffusers, and approved Hugging Face speech-runtime
gaps. It is a durable product roadmap rather than a claim that every upstream
pipeline is already runnable.

The reviewed Diffusers installation is pinned to commit
`bb56997d4b7e87f0743f26a612f49ec4e7ce7213`. The latest-upstream check on
2026-08-13 found [Diffusers v0.39.0](https://github.com/huggingface/diffusers/releases/tag/v0.39.0)
as the latest tagged release (release commit
`a3608b512ed7248499a44c61d954965ed9bdae4d`) and
`bb56997d4b7e87f0743f26a612f49ec4e7ce7213` as the latest `main` commit. The
comparison inventory and executable dependency now use that reviewed immutable
snapshot. Re-run the inventory before changing the Diffusers pin or marking a
gap complete.

### Pin delta admitted 2026-08-13

The admitted pin is 73 commits after the prior
`13a7bee4878d62fccc8d25f97e480e68de96fa03` snapshot. Its product-relevant
surface consists of Krea2 Modular support, MiniMax H3, LTX-2.5 plus its final
`LTX25AutoBlocks` rename, Wan-Animate-2, SDNQ support, JAX/Flax removal, and
core changes to component management, group/automatic offload, split-device
deduction, dtype naming, LoRA scaling/bookkeeping, GGUF dequantization, and
custom-block required-input propagation. The remaining commits are tests,
documentation, training/examples, CLI work, or model-specific fixes outside
the currently admitted execution surface.

The update is isolated in backend commit `5ee9e1d`; no client change was
required. Regenerating the existing 20-class no-weight Modular contract
inventory produced no structural drift beyond the pin identity. MoDiff carries
the upstream required-custom-input fix into its adapted schema helper and adds
an exact regression test. The full backend suite passed at the proposed pin
(`1312 passed, 3 skipped, 2957 subtests`) in an isolated reviewed
Transformers/PEFT test environment. The repaired clean CPU base contains 61
application packages, keeps Transformers and PEFT absent, reports the exact
new VCS identity, passes package validation, and is preflight-ready with an
exact requirements receipt. No weights or media were downloaded.

LTX-2.5 reuses the standard `LTX2Pipeline` family rather than adding a
model-named standard pipeline, but it adds execution behavior that must be
reviewed explicitly:

- the immutable `Lightricks/LTX-2.5-Diffusers` artifact and its distinct
  distilled `transformer/`, full/SFT `transformer_full/`, latent upsampler, and
  stage-2 distilled-LoRA receipts;
- the reference distilled sigma schedules and both supported two-stage
  generation variants, without substituting a generic step-count schedule;
- `LTX2DurationHead`, the optional Gemma-4 prompt-enhancement component, and
  their bounded/explicit controls (no discovery-time model download);
- `LTX2VideoDiffusionDecoderModel` and
  `LTX2VideoDiffusionDecodePipeline`, including the production-resolution
  NATTEN dependency and video/audio latent handoff contract; and
- the new `LTX2ModularPipeline` and `LTX25ModularPipeline` exports, with
  `LTX2AutoBlocks`/`LTX25AutoBlocks` covering text-to-video,
  image-to-video, condition-to-video, and IC-LoRA/in-context selection.

MiniMax H3 was already present in the prior inventory from upstream commit
[`f53d552`](https://github.com/huggingface/diffusers/commit/f53d552) and remains
post-pin. Its Phase 6 item now records the three separate joint video-and-audio
workflows: text-only `t2va`, first/last-keyframe `fl2va`, and omni-reference
`ref2va`. `t2va`/`fl2va` use the repository's `transformer/` partition;
`ref2va` uses `transformer_ref/`. These are future generic video+audio task
contracts, not permission to add a MiniMax-named node or enable an unqualified
artifact.

## How to update this tracker

- Use `[x]` only after every required backend, client, test, live-proof, and
  asset item for that checkbox is complete.
- Record the backend and client commit or pull-request references in the
  completion ledger. Do not create empty commits in either repository; a
  backend-only segment instead records the compatible client gate that passed.
- Keep a template hidden or explicitly `qualification_pending` until its live
  output and public assets meet the publication contract.
- Update this file in the same change that completes or reschedules a segment.
- Distinguish contract, mocked, tiny-model, live-output, and Gallery-asset proof.

## Completed groundwork

- [x] Inventory the current backend profiles, Auto requirements, nodes, graphs,
  and Gallery manifest.
- [x] Compare the pinned Diffusers revision with the reviewed upstream snapshot.
- [x] Identify the initial 18 missing official Modular pipeline classes and 69
  missing standard pipeline families listed in the appendices.
- [x] Re-audit upstream through 2026-08-12 and append the two newly exported
  LTX2/LTX2.5 Modular classes, bringing the current Modular gap inventory to
  20 without changing the 69-family standard-pipeline inventory.
- [x] Run the pre-change backend baseline: 627 tests and 274 subtests passed;
  Ruff, dependency validation, and backend preflight passed.
- [x] Confirm the following owner decisions:
  - Official libraries maintained by Hugging Face may be added as reviewed
    model runtimes; Transformers speech-to-text is the first planned use.
  - Transformers and other workflow-specific Hugging Face model runtimes are
    optional and are not installed by the base application installer.
  - No hosted inference provider or browser-side model runtime is approved.
  - MoDiff will not support Mellon's configuration filename or schema.
  - `modiff_pipeline_config.json` is the only MoDiff dynamic-node sidecar.
  - Release media should be generated on a separate qualification machine.
  - No command or model run on the current development machine may exceed 40
    minutes.

## Non-negotiable architecture decisions

### Generic nodes, explicit adapters

Nodes represent tasks and media contracts, not model names. Existing generic
image, audio, and video nodes should be extended before adding another
`NodeBase` class. New generic contracts are allowed for genuinely different
semantics such as unconditional image generation, perception maps, 3D
artifacts, diffusion text, and speech recognition.

Model-specific behavior belongs in a declarative execution specification that
records:

- exact `(modelType, mode)` identity;
- loader and generator module/action;
- upstream pipeline class or approved Hugging Face runtime class;
- required, optional, and aliased inputs;
- normalized output contract;
- immutable model, adapter, and auxiliary revisions;
- dtype, quantization, placement, and offload constraints;
- template coverage and qualification state.

Do not pass arbitrary form fields to a model and hope its call signature accepts
them. Generic means a stable user contract backed by validated adapters.

The frontend follows the same rule. It should render a generic task contract
such as `control_image` from backend-declared fields and constraints, without a
Qwen-only or Flux-only graph-building branch. The backend execution
specification selects standard versus Modular loading, maps parameter aliases,
and rejects unsupported combinations. Curated templates remain model/task
specific only as data: they carry reviewed artifacts, defaults, prompts, and
evidence while reusing the same generic node and form implementations.

### Approved Hugging Face execution boundary

MoDiff may execute models through official libraries maintained by Hugging
Face, including Diffusers, Transformers, and future reviewed Hugging Face
libraries. This broadens the model-runtime boundary, not the graph or trust
boundary:

- MoDiff's existing graph executor remains the only graph executor;
- every new library and task receives an explicit generic node/adapter contract;
- executable dependencies and model/auxiliary artifacts are reviewed and
  immutably pinned where the source supports revisions;
- safetensors is preferred and unsafe deserialization is an explicit reviewed
  exception;
- `trust_remote_code` is never enabled implicitly;
- no hosted Inference Provider or browser-side model runtime is added;
- arbitrary Hub Python is not made trusted merely because it is stored on the
  Hugging Face Hub;
- workflow-specific Hugging Face libraries are installed through an explicit,
  reviewed first-use action instead of the base application installation;
- file, network, input-size, output-size, resource, and cleanup limits remain in
  force.

Library ownership alone does not prove that every task or model is supported.
Each exact task still progresses through contract, fixture, live, and Auto
qualification states. Transformers automatic speech recognition and speech
translation are the first planned non-Diffusers tasks. They use a generic model
loader and transcription node, not Whisper-specific nodes, with bounded audio
and text contracts.

### Lazy optional Hugging Face runtimes

The base application environment must not directly depend on Transformers or
another library needed only by particular templates. An execution specification
declares its required runtime packages. When a user first tries to run a
template or workflow whose package is absent, MoDiff blocks execution and
offers an explicit install action. Merely opening a template, discovering
nodes, or requesting an Auto plan must not download or install packages.

After confirmation, the backend stages the reviewed package set, validates it
in a fresh process, activates it atomically, restarts when required, and retains
the prior environment for rollback. The client shows download, validation,
activation, restart, failure, and rollback states. A failed or declined install
leaves the workflow unchanged and Expert-visible with a concrete missing-runtime
reason.

Transformers and PEFT are currently direct project dependencies, and PEFT has
an unconditional Transformers dependency. They therefore move out of the base
environment together in one compatibility segment. Do not remove either until
registry discovery, preflight, existing Diffusers workflows, optional
installation, restart, and rollback all pass from a clean base installation.

### MoDiff-only dynamic configuration

MoDiff does not read `mellon_pipeline_config.json` and will not add a fallback
for it. Dynamic Modular configuration uses canonical upstream metadata such as
`modular_model_index.json` and `modular_config.json`. Optional MoDiff UI fields
and defaults use `modiff_pipeline_config.json` only.

The currently curated `diffusers/FLUX.2-klein-4B-modular` example is not a
valid MoDiff dynamic-block example because its reviewed revision does not
publish `modiff_pipeline_config.json`, and an auxiliary model reference is not
immutably pinned. Remove it from the curated selector and bundled graph until a
reviewed repository satisfies the MoDiff contract. A user-selected repository
without the MoDiff sidecar receives an actionable unsupported-config error.

### Auto is fail-closed

Auto may run only an exact registered and qualified `(modelType, mode)` recipe.
An installed artifact, a pipeline class name, a mocked test, or a successful
lighter task is insufficient proof. Unknown or unqualified combinations remain
visible only in Expert mode with a specific reason.

### Local 40-minute ceiling

Every local command, download/load/inference job, and test batch must have a
wall-clock timeout of at most 40 minutes. Model smokes should be designed for a
30-minute expected maximum. Request graceful cancellation by 35 minutes and
retain 5 minutes for cleanup and diagnostics. At the hard timeout:

1. cancel the run;
2. release managed model and accelerator resources;
3. record the candidate as not locally qualified, not as failed upstream
   support;
4. move the workload to the remote qualification queue;
5. do not retry the unchanged recipe on this machine.

Permitted local live candidates are initially limited to small DDPM/DDIM,
Consistency Models, short low-resolution SD/LCM/PAG, small Marigold, and
Whisper Tiny/Base with a short audio fixture. Release assets are still produced
on the remote machine by default. Video, long audio, AudioLDM2 quality/TTS,
large image models, and long-form workflows are remote-only.

## Commit and asset handoff contract

Every segment ends at a clean source-control boundary.

1. **Backend source commit:** nodes/adapters, execution specifications, API
   changes, graph contracts, focused tests, and documentation.
2. **Client source commit:** typed capability handling, Studio profile/form,
   graph bridge, template metadata, readiness UX, and unit/browser tests.
3. **Integration gate:** complete backend and client checks against the paired
   commits. No generated media is required at this point; public Gallery
   activation remains blocked.
4. **Remote qualification:** the other machine checks out the exact two commits,
   installs the reviewed profiles, runs the model, and records the model and
   dependency revisions, graph hash, settings, output checks, runtime, and peak
   memory.
5. **Asset publication:** reviewed media is uploaded to the public Hugging Face
   Dataset. Generated images, audio, and video are not committed to Git.
6. **Activation commits:** the client commits the immutable Dataset revision,
   SHA-256 manifest, rights/provenance record, review, and Gallery status. Its
   generated `dist/` is mirrored into backend `web/` using the documented
   process; minified files are never edited manually.

An integration may merge before remote assets exist only when its UI says
`qualification_pending`, Auto is disabled, and no public Gallery entry implies
live proof. Asset activation is a separate committable segment.

Broad repeatable entries such as a model-family template batch must use one
family/mode per paired commit. Add suffixed ledger rows such as `P2.2a` and
`P2.2b`; do not combine unrelated families merely because they share a phase.

## Known integration defects that set the initial order

### Flux Modular falsely claims ControlNet

Before the P0.1 working-tree fix, the backend capability table said
`FluxModularPipeline` supported
`control_image`, while its node specification explicitly sets `controlnet` to
`None`. The pinned upstream `FluxAutoBlocks` exposes `text2image` and
`image2image`, not a ControlNet workflow.

A user could encounter the contradiction in two ways:

1. A consumer of `/model_capabilities` sees `control_image` as runnable for
   `FluxModularPipeline`, even though the graph runtime cannot construct it.
2. In a generic Modular graph, connect a Modular `ControlNet` node and switch
   the signalled model type to `FluxModularPipeline`. The node-definition update
   receives the `None` specification and removes its control image, model, and
   scale fields; the Flux denoise definition also lacks `controlnet_bundle`. The
   graph becomes unwireable and can retain stale edges. A stale imported or
   hand-edited graph that retains those fields reaches execution and tries to
   iterate `node_config["params"]`, causing a `NoneType` failure. A full graph
   may load large Flux components before reaching that failure.

There is no checked-in `FluxModularPipeline:control_image` Studio template, so
the normal Gallery path never executes this combination. The current bundled
client also discards `experimentalCapabilities`, so it does not create a normal
Flux Modular Control button from this bad entry. The Flux Canny and Flux Depth
control templates use standard Diffusers pipelines and are not affected.
Exactly one checked-in graph contains the Modular ControlNet node:
`qwen-image-modular-pipeline/control-image.json`. It signals Qwen, whose
configuration is present, and fails only if a user manually changes that graph
to Flux.

Git history shows that the initial Modular import commit `346c203` inherited
both the Flux option in the generic ControlNet signal map and the explicit
`controlnet: None` marker from the recorded Mellon baseline. The best-supported
interpretation is that the signal map named known models so the generic node
could reconfigure or hide itself; it was not itself intended to declare
support. Its missing symmetric execution guard was still a defect. Commit
`76bbafe` later added the public experimental capability claim. Existing tests
check registry exports and working generic contracts, but do not enforce that
every advertised mode has a non-null node specification and an upstream
workflow. The history does not record why the later claim was added; confusing
standard Flux Control pipelines with Modular Flux support is plausible but is
only an inference.

P0.1 removes the false capability and makes both dynamic-node update and stale
graph execution return the same actionable unsupported-workflow error. The
generic signal map remains only a reconfiguration mechanism; it is not treated
as proof of a runnable workflow.

### The curated DynamicBlock example is an incomplete Mellon-to-MoDiff migration

MoDiff does not currently request or parse `mellon_pipeline_config.json`.
Commit `346c203` originally used Diffusers' inherited `MellonPipelineConfig`
helper with `YiYiXu/FLUX.2-klein-4B-modular`. Commit `57b9bb` introduced
`MoDiffPipelineConfig`, renamed the sidecar to `modiff_pipeline_config.json`,
and documented that legacy filenames are unsupported, but it did not replace
the example. Commit `76bbafe` switched the curated option to the pinned
`diffusers/FLUX.2-klein-4B-modular` repository, which still has Mellon UI
metadata rather than a MoDiff sidecar.

The current revision-forwarding unit test mocks the configuration loader, so it
cannot discover that the real pinned repository lacks the requested file. This
is fixed by removing/replacing the curated example and adding repository-layout
contract coverage, not by restoring Mellon compatibility. Attribution in the
source-provenance map remains unchanged because provenance is not a runtime
compatibility promise.

## Model-dependent node and workflow audit

The 2026-08-07 follow-up audit checked every registered Modular pipeline against
the pinned Diffusers classes and block definitions without loading model
weights. The generic node class is shown at the top of each column; `yes` means
that the registered MoDiff specification has a non-null action contract, not
that a model has completed live qualification.

| Registered Modular model | Encode Prompt | Image Embeddings | Image Encode | Denoise | Decode Latents | ControlNet |
| --- | --- | --- | --- | --- | --- | --- |
| `StableDiffusionXLModularPipeline` | yes | no | yes | yes | yes | yes |
| `QwenImageModularPipeline` | yes | no | yes | yes | yes | yes |
| `QwenImageEditModularPipeline` | yes | no | yes | yes | yes | no |
| `QwenImageEditPlusModularPipeline` | yes | no | yes | yes | yes | no |
| `QwenImageLayeredModularPipeline` | yes | no | yes | yes | yes | no |
| `FluxModularPipeline` | yes | no | yes | yes | yes | no |
| `FluxKontextModularPipeline` | yes | no | yes | yes | yes | no |
| `Flux2KleinModularPipeline` | yes | no | yes | yes | yes | no |
| `ZImageModularPipeline` | yes | no | yes | yes | yes | no |
| `WanModularPipeline` | yes | no | no | yes | yes | no |
| `WanImage2VideoModularPipeline` | yes | yes | yes | yes | yes | no |

`DummyCustomPipeline` is deliberately absent from the matrix because its
actions must be discovered from the reviewed `modiff_pipeline_config.json`; it
must never inherit a built-in model's actions. An absent action is an explicit
unsupported model/action pair. The frontend must hide or disable it and the
backend must reject stale or hand-edited graphs before loading weights.

The custom path currently uses one mutable `DummyCustomPipeline` class and one
registry slot. Loading a standard model resets that slot, while loading a
second custom repository overwrites the first. Runtime payloads restore the
first repository's ID, revision, and trust flag but not its matching sidecar,
so a later node can resolve the wrong or empty action contract. Custom support
therefore needs an immutable per-`(source, repository, revision, explicit trust
choice, sidecar hash)` identity and isolated binding before it can share the
built-in completion claim.

The audit found the same failure family as the former Flux ControlNet defect in
the other five dynamic nodes. Their definition-update handlers can silently
clear fields for a non-null-to-null model switch, while stale execution can
dereference the missing specification. They also remove connector fields from
the returned parameter mapping in place. Built-in mappings happen to be
reconstructed, but a dynamic custom mapping can be permanently changed by the
first update. Finally, runtime component payloads are not always reconciled
against the node's selected pipeline class, so components from two model types
can reach the wrong action specification.

The surrounding generic nodes also contain undeclared compatibility rules:

- `Denoise` has a class-name list that controls whether width and height are
  shown for image-conditioned models.
- `ModelsLoader` assumes a text encoder exists and special-cases the Wan I2V
  image encoder instead of deriving required and optional components.
- `AutoModelLoader` describes a component and repository but not the owning
  Modular pipeline contract. A graph built entirely from standalone component
  loaders therefore cannot recover its exact action schema after transient UI
  signals are lost. The execution specification must supply that identity; it
  must not be guessed from a repository name.
- `Layers` publishes a hard-coded family map and omits several registered
  models; missing entries must be researched rather than copied from a nearby
  architecture.
- `Guider` is task-generic but does not declare model compatibility. The pinned
  Diffusers revision has two guiders that MoDiff does not expose, and latest
  `main` adds a third. Flux Modular specifications currently show a guider
  input even though the upstream Flux Modular pipelines have no guider
  component.
- `Scheduler` offers every registered scheduler to every model without an
  exact compatibility contract.

The standard task nodes have related adapter-specific gaps even though they do
not create model-specific node classes:

- Image loading validates pipeline class and mode, but individual Generate,
  Edit, Inpaint, and Control actions do not revalidate the connected adapter's
  allowed mode. Changing only the pipeline-class field can also retain the Flux
  Schnell repository default for Z-Image, Flux2 Klein, Flux Fill, Flux Control,
  Flux Kontext, or Flux Redux. Auto and templates normally overwrite the
  repository, which hid this manual-workflow defect.
- Video generation does revalidate modes, but its loader's final branch assumes
  FramePack and an untagged pipeline falls back to Wan VACE. Both must fail
  closed when exact recovery is impossible.
- Audio loading knows ACE-Step versus Stable Audio, but generation does not
  retain and revalidate the selected mode/task contract. Stable Audio can show
  ACE-specific task choices, while its own `text2audio` task is absent from the
  visible choices.

The raw Expert canvas is already structurally generic: it renders `/nodes`
metadata and executes backend `onChange` and `onSignal` actions. Studio graph
authoring is not. Its model union, profiles, role-to-node mapping, topology,
edge recipes, initialization order, form bindings, pipeline classes, and many
readiness/resource messages are selected by Flux, Qwen, Wan, Z-Image, audio,
or video branches. `/model_capabilities` schema v2 can say whether a pair is
runnable, but it cannot yet describe the nodes, handles, bindings, or ordered
dynamic actions needed to build that pair's graph.

The target contract is a backend-owned `studioExecutionSpec`, keyed by exact
`(modelType, mode, executionProfileId)`. It references existing `/nodes` keys
and declares stable roles, nodes, edges, accepted handle aliases, form fields
and bindings, initialization actions, normalized inputs/outputs, a schema
version, and a content hash. It is configuration for the existing MoDiff graph
executor, not a second graph representation or executor. The client validates
every referenced node, parameter, and handle against `/nodes`, then materializes
the same visible editable graph. Templates carry an execution-spec reference
and reviewed overrides instead of hidden model-specific code.

### Upstream workflow coverage discovered by the node audit

| Family | Pinned upstream workflow surface | Current MoDiff gap or mismatch |
| --- | --- | --- |
| SDXL Modular | text-to-image, img2img, and inpaint, each with ControlNet, ControlNet Union, IP-Adapter, and combined variants (18 workflows) | All 18 pinned workflows now have exact generic action/state truth for generator continuation, typed mask/masked latents, crop overlay, exact VAE/ControlNet provenance, and process-local adapter mutation/embedding provenance. The four high-level base modes, including inpaint, are published only as `contract_only`; the combined variants remain manual generic-graph compositions rather than new model-named modes. Multi-ControlNet and multiple-IP-Adapter variants are outside the pinned 18-workflow contract. |
| Qwen Image Modular | text-to-image, img2img, inpaint, plus ControlNet versions of all three | Direct text-to-image and Modular control text-to-image are exposed. The generic main-VAE image/mask/overlay route, ControlNet generator/provenance chain, and exact internal combined img2img/inpaint state-flow contracts are implemented. Combined modes remain unadvertised and still require profile/template/live qualification. |
| Qwen Edit Modular | image-conditioned and image-conditioned inpainting | Edit is exposed. The generic VAE/denoise/decode generator, mask, and overlay route is implemented contract-only; Modular inpaint exposure and qualification remain pending. The separately registered standard inpaint/outpaint path is unaffected. |
| Qwen Edit Plus | upstream block sequence, without an upstream workflow map | Core actions and the generator-only VAE/denoise/decode route exist. Inpaint state is rejected; multi-image input cardinality and field normalization still need contract tests before broader exposure. |
| Qwen Layered | upstream block sequence, without an upstream workflow map | Core actions now expose the pinned shared 640/1024 source resolution plus text-only English-prompt and bounded maximum-sequence controls. Broader exposure remains tied to the reviewed fixed-block contract and later qualification. |
| Flux Modular | text-to-image and img2img | Current modes match; Modular ControlNet remains unsupported. |
| Flux Kontext Modular | text-to-image and image-conditioned | Registered internally but no public Modular execution profile/specification. |
| Flux2 Klein Modular | text-to-image and image-conditioned | Registered internally; current experimental metadata points at a standard pipeline and advertises edit semantics without publishing a Modular execution path. |
| Z-Image Modular | text-to-image and img2img | Text-to-image only is declared; existing generic latent/strength fields make img2img a small contract/profile gap, still requiring qualification. |
| Wan I2V Modular | image-to-video and first/last-frame video | The existing image-to-video mode has its exact split-action typed edges and opaque route. The distinct official FLF checkpoint is now pinned immutably and admitted as a reviewed repository variant of the same generic Models Loader; action admission requires the I2V/FLF input shape to match that exact loader publication. Public FLF profile/template promotion and live execution remain pending. |
| Wan T2V Modular | canonical text-to-video block sequence | The declared text-to-video action matches the available block sequence. |

This table is an admission inventory, not permission to advertise every
upstream workflow. Support is added only when all required node actions,
parameter adapters, execution specification, tests, and proof level agree.

### Missing actions inside families MoDiff already supports

These are action/adapter gaps within existing families, separate from the 69
entirely missing standard families in Appendix B. They should extend generic
image, video, and audio task nodes; none justifies a model-named node.

| Family | Existing generic coverage | Confirmed pinned classes/actions not yet covered or exposed |
| --- | --- | --- |
| SDXL | Modular text/image/control/inpaint high-level modes and standard text/img2img/inpaint adapters published contract-only; exact ordinary/bounded single-ControlNet-Union and reviewed single-IP-Adapter generic state flows | instruct-pix2pix; live Modular qualification, multi-ControlNet, multiple IP-Adapters, and templates for combined actions |
| Qwen Image | standard text-to-image and Edit inpaint/outpaint; contract-only standard img2img, inpaint, Edit, and Edit Plus; Modular control text-to-image and edit paths | standard ControlNet, ControlNet inpaint, and Layered adapters; Modular img2img/inpaint combinations |
| Z-Image | standard and Modular text-to-image plus contract-only standard img2img/inpaint | standard ControlNet, ControlNet inpaint, and Omni adapters; Modular img2img exposure |
| Flux | text/image edit, fill, base control, ControlNet, Kontext, Redux, contract-only img2img/inpaint/Kontext-inpaint, and exact true-CFG forwarding | control-img2img, control-inpaint, ControlNet-img2img, and ControlNet-inpaint |
| Flux2 | Klein text/image edit and multi-reference plus contract-only Klein inpaint | KV and full Flux2 after artifact/runtime review |
| Wan | five profiled standard video adapters, Modular T2V/I2V, and contract-only Wan 2.2 T2V/Animate adapters | first/last-frame profile; the three Expert-only VACE video/reference/color modes; a truthful VACE first-frame adapter that synthesizes the required video-and-mask state; live qualification for Animate |
| LTX/LTX2 | profiled LTX condition modes plus contract-only long-prompt I2V and LTX2 condition adapters | latent upsample; LTX2 in-context, HDR, and latent-upsample actions; post-pin LTX-2.5 distilled/full and two-stage recipes, duration head, Gemma-4 prompt enhancer, diffusion decoder, and LTX2/LTX2.5 Modular pipelines; live execution qualification for long/LTX2 |
| Hunyuan Video | FramePack adapter published contract-only | base text-to-video, image-to-video, and SkyReels image-to-video; live FramePack qualification |
| Audio | ACE-Step plus Stable Audio published contract-only | live Stable Audio qualification; unsupported ACE `extract`, `lego`, and `complete` choices remain hidden until their missing inputs exist |

All entries begin `contract_only`. Real output, resource envelopes, Auto
qualification, and Gallery publication remain separate remote-machine work.

## Test and evidence ladder

Each phase below names its applicable levels.

1. **Static/contract:** registry closure, class and signature availability,
   immutable revisions, safetensors/trust policy, schema and manifest checks.
2. **Mocked/tiny:** fake pipeline calls, input aliasing, output normalization,
   Modular state/component flow, and tiny upstream fixtures without large
   downloads.
3. **Integrated backend/client:** HTTP contracts, Studio graph application,
   Auto/Expert gating, stale-request handling, and mocked browser flows.
4. **Local short live:** only an allowlisted workload that is expected to finish
   inside 30 minutes and is forcibly bounded at 40 minutes.
5. **Remote qualification:** exact normal recipe on the target qualification
   hardware, producing attributable output and a resource receipt.
6. **Gallery publication:** rights review, inventory, hash verification,
   anonymous remote verification, activation, and release acceptance.

Required backend gate:

```powershell
uvx --from ruff==0.12.7 ruff check . --select E9,F
uv pip check --python .venv/Scripts/python.exe
.venv/Scripts/python.exe -m modiff.preflight --json --check-port 8088 --fail-on-error
.venv/Scripts/python.exe -m pytest -q
```

Required client gates for public-contract or UI changes:

```powershell
npm run check
npm run check:ui
```

Template and Gallery changes also require:

```powershell
npm run gallery:verify
npm run gallery:coverage
npm run workflows:verify
npm run test:asset-storage
```

Asset activation additionally follows the complete release gate in the client
`docs/template-gallery-assets.md` guide, including `npm run check:acceptance`,
`npm run release:assets:gate`, `npm run release:template:qualify`, and
`npm run release:resource:qualify` on the external qualification host.

## Phase 0 — Auto correctness and registry closure

Priority: immediate. Hardware: CPU only. Assets: none.

### Committable segments

- [x] **P0.1 Exact pair validation and Flux truth fix**
  - Backend: require an exact `(modelType, mode)` entry; remove
    `control_image` from `FluxModularPipeline`; make a `None` node specification
    an explicit unsupported result in both node update and execution; never
    treat membership in a generic signal map as a capability declaration.
  - Client: do not offer a backend mode absent from the exact capability; show
    the backend reason and preserve Expert access only for structurally valid
    graphs.
  - Tests: unknown pair, wrong-mode pair, Flux control capability absence,
    dynamic node update, stale imported graph, and mocked Studio mode gating.
  - Status 2026-08-07: implementation and source gates are complete in paired
    backend `91c9a36` and client `28b12b7`. Auto now requires
    the pair in both its requirements and execution-profile registries, rejects
    stale plan/form identity mismatches at execution, and cannot be promoted by
    installed artifacts, history, or client-supplied proof. The client treats
    schema-v2 `runnableModes` (including `[]`) as authoritative and submits the
    exact model and mode in runtime hints.
  - Evidence: backend Ruff, dependency validation, preflight, and the complete
    suite passed (`639 passed, 318 subtests passed`). Client `npm run check`
    passed, and the focused mocked-browser regression passed (`1 passed`). The
    complete Windows browser run executed all 75 cases: the 74 functional cases
    passed, while one unrelated layout-snapshot case reported only the two
    absent Win32 JSON baselines; the generated baselines were removed. No model,
    media, or Gallery assets were downloaded or generated.
- [x] **P0.2 Explicit resource-plan targeting**
  - Backend: put loader module/action and execution path in every specification;
    remove class-name substring routing; fail if a plan changes zero matching
    loaders.
  - Client: verify the returned target matches the visible graph before applying
    a plan; surface a mismatch instead of enabling Run.
  - Tests: Qwen Modular, Qwen Edit Plus, Qwen Layered, Wan Modular, mixed-loader
    graphs, stale candidate IDs, and zero-update plans.
  - Status 2026-08-10: paired backend `8fb2cb9` and client `c3e8a17` make the
    reviewed execution profile the exact Auto authority for loader module,
    loader action, execution path, pipeline class, model type, mode, and pinned
    artifact. Plans, selected candidates, candidate lists, retries, history, and
    runtime hints retain and cross-check that identity. The backend targets only
    exact executable-path loaders, rejects missing, ambiguous, stale,
    cross-profile, disconnected, and zero-target plans, and emits bounded
    non-echoing failures. The client validates bounded schema-v2 responses,
    requires an exact selected/list receipt, checks an enabled managed loader
    before readiness, mutation, and submission, and exposes a mismatch instead
    of enabling Run. Z-Image Auto uses its reviewed direct-image adapter while
    its separate Expert Modular capability remains unchanged.
  - Evidence 2026-08-10: the complete backend gate passed 1113 tests with 4
    skips and 1688 subtests; the focused independent P0.2 replay passed 172
    tests and 330 subtests. Ruff E9/F, dependency validation, preflight, and
    diff checks passed. The client `npm run check` and all 86 mocked Studio
    browser cases passed; focused contracts passed 76/76 and an independent
    critical-browser replay passed 7/7. The production bundle is 523003 bytes
    gzip, 133 bytes inside the stricter 523136-byte safety target. The mirrored
    client matched all 26 generated files byte-for-byte, and a fresh backend
    served `/`, `/assets/index.js`, `/health`, and `/runtime/status`. These are
    CPU/static/unit/contract/mocked-browser and local HTTP results: no model or
    Gallery asset was downloaded, and no model or media output was generated.
- [x] **P0.3 Canonical execution-spec registry and generic node closure**
  - Deliver this as the following independently committable, CPU-only paired
    segments. None downloads a model or generates an asset.
  - [x] **P0.3a.1 Registered Modular dynamic action safety**
    - Backend: add one action-contract resolver used by Encode Prompt, Image
      Embeddings, Image Encode, Denoise, Decode Latents, and ControlNet; derive
      their fields from a defensive copy of the selected model specification;
      reject absent actions consistently during definition update and stale
      execution; reconcile every connected component's `model_type` against
      the selected pipeline rather than letting cached node state win.
    - Client: keep the generic `/nodes` renderer and dynamic field-action path;
      add a mocked browser matrix proving that switching the connected model
      changes fields without a model-name branch, removes unsupported actions,
      and reports stale imported edges cleanly.
    - Tests: every registered pipeline by every dynamic action; supported and
      unsupported updates; selected-versus-connected identity mismatch; unknown
      class; repeated custom-sidecar updates without registry mutation; existing
      Auto exact-pair behavior unchanged.
    - Status 2026-08-07: implementation and source gates are complete for the
      eleven registered built-in classes in paired backend `91c9a36` and client
      `28b12b7`. The backend now uses
      one immutable contract resolver and one actionable model-type resolver
      across all six nodes. Runtime recovery rejects mixed selected/connected
      identities, while SDXL's valid bundle-only ControlNet contract remains
      supported. ControlNet now forwards any signalled model identity instead
      of maintaining a class allowlist. The client test uses a synthetic generic
      task/model contract and proves backend definitions remove and restore
      executable fields with no product model-name branch.
    - Evidence: the focused backend run passed (`56 passed, 92 subtests
      passed`); the complete backend suite passed (`649 passed, 402 subtests
      passed`), along with Ruff, dependency validation, and preflight. Client
      `npm run check` passed, the two focused dynamic-definition browser tests
      passed, and the complete mocked Studio run passed all 75 functional cases;
      its only failure was the pre-existing absent Win32 JSON baselines for one
      layout snapshot. `check:ui` likewise reached the pre-existing absent
      Win32 PNG baselines. All six generated baseline candidates were removed.
      No model, media, or Gallery asset was downloaded or generated.
  - [x] **P0.3a.2 Custom Modular contract identity isolation**
    - Backend: replace the global mutable Dummy configuration with an immutable
      contract identity and bounded cache keyed by source (`hub` or `local`),
      repository, revision, explicit remote-code trust choice, the verified
      `modiff_pipeline_config.json` hash, and a bounded executable-metadata
      manifest hash. Resolve the exact sidecar bytes without network access or
      upstream construction. Runtime recovery must restore the matching
      declarative contract without one standard or custom loader resetting
      another graph. Hub recovery requires an exact cached commit; local
      recovery rejects containment and executable symlink escapes. Keep
      per-identity custom bindings out of the public built-in enumeration and
      use locked, defensive snapshots.
    - Client: carry the same backend-issued opaque custom contract identity in
      two generic carriers without interpreting repository names or identity
      fields: a hidden loader value that is persisted, exported, and included in
      run hashes, plus transient output signals that drive connected dynamic
      fields. Signal values remain non-durable. Rebase stored nodes onto current
      backend action metadata and hidden contract fields so legacy graphs can
      acquire the identity without a model-specific migration.
    - Security boundary: custom pipeline and Dynamic Block execution is
      `contract_only` in this segment, even with `trust_remote_code=false`.
      The pinned upstream constructor imports the installed library named by a
      repository-controlled component `type_hint` before MoDiff can approve
      it. `trust_remote_code=true` and non-boolean lookalikes fail before cache
      reuse or upstream calls. Sidecar callbacks are declarative-only, imported
      client actions remain inert until rebased from `/nodes`, and
      `/fields/action` validates the live module, node, field, and callback.
      Built-in Modular execution is separately bound to the registered default
      repository, cataloged commit, pipeline class, and installed component
      type hints. The generic standalone component loader resolves a reviewed
      Diffusers model class from bounded `config.json` rather than letting
      repository `model_index.json` select an installed package.
    - Tests: custom A, standard B, then A again; interleaved custom A/B nodes;
      restart/recovery from self-describing inputs; sidecar hash/revision
      mismatch; source/trust/execution-ID tampering; strict boolean validation;
      missing, malformed, duplicate-key, or oversized local sidecar; Hub/local
      shadowing and local symlink escape; no network during recovery; concurrent
      registry access and failed-load cache isolation; repository-class
      masquerading; cache-mutated component hints; standalone component
      category/class mismatch; arbitrary installed-package dispatch; and
      authoritative field-action dispatch. Client tests cover opaque
      propagation, disconnect clearing, hidden-value graph round-trip and
      run-hash participation, legacy/empty-registry inert hydration and live
      rebasing, source/revision commit actions, and switching back to a normal
      transient signal. Assets and model weights: none.
    - Status 2026-08-09: implementation and all source, contract, browser, bundle,
      and fresh-backend HTTP gates are complete in paired backend `91c9a36` and
      client `28b12b7`. The identity is a content checksum used for recovery and run
      hashing, not authorization. Its executable-metadata manifest detects
      bounded Python and loader-config drift but deliberately does not claim
      atomic custom-code execution or model-weight proof. Executable custom
      pipelines and Dynamic Blocks moved to the reviewed P1.1 admission contract
      rather than weakening this boundary.
    - Evidence: the combined focused backend matrix passed (`166 passed, 180
      subtests passed`); the complete backend suite passed (`709 passed, 482
      subtests passed`), with only the reviewed upstream `torch_dtype`
      deprecation warning. Ruff `E9,F`, dependency validation (`78 packages
      compatible`), compile checks, diff checks, and preflight (`ready: true`)
      passed. Client `npm run check` passed, including formatting, lint, type
      checking, all unit tests, the production build, and the bundle budget
      (`523172 / 523264` gzip bytes). The three focused mocked-browser dynamic
      contract regressions passed. The exact built `index.js` was mirrored into
      the backend without touching backend-owned Gallery media; a fresh backend
      served `/` and `/assets/index.js` with HTTP 200 and the expected 1,303,875
      byte asset. No model, media, Gallery asset, Transformers installation,
      network model download, or GPU execution occurred.
  - [x] **P0.3b Standard adapter fail-closed closure**
    - Backend: choose the official default repository after pipeline-class
      changes; revalidate image actions against adapter modes; make video loader
      dispatch and untagged-pipeline recovery explicit; preserve and validate
      audio mode/task/input identity; add permanent direct-profile-to-adapter
      closure tests.
    - Client: filter generic action and task choices from backend declarations,
      with no Flux, Wan, ACE-Step, or Stable Audio selection branch; show an
      unsupported-contract error for stale graphs. Remove ModelSelect's
      client-owned family inference and repository-name matching so the live
      backend `fieldOptions.filter` contract is authoritative for both Hub and
      local choices, including future model families.
    - Tests: fake pipelines only, including manual class-only changes, wrong
      image action, unknown video adapter, Stable Audio task filtering, ACE
      mode/task disagreement, every direct profile/class/mode tuple, and a
      synthetic future-family ModelSelect option admitted solely by backend
      metadata.
    - [x] **P0.3b.1 Registry and direct-profile closure:** declare ordered modes,
      reviewed primary/compatible repositories, and exact load/execute handlers;
      permanently test every public direct profile maps to an adapter and every
      advertised mode is implemented. Expert-only adapters need not be promoted
      into Studio. CPU/static tests only; assets: none.
    - [x] **P0.3b.2 Generic image action identity:** resolve backend-managed
      defaults after a class change while preserving explicit Hub/local choices;
      tag class, mode, and repository even on cache hits; validate Generate,
      Edit, Inpaint/Outpaint, and Control actions before upstream execution; and
      publish generic field filters/contracts from the backend. Remove
      `FluxControlNetPipeline` from selectable claims until the generic loader
      can supply its required ControlNet component. Fake pipelines only; assets:
      none.
    - [x] **P0.3b.3 Generic video dispatch and recovery:** replace FramePack and
      Wan-VACE catch-all fallbacks with exact handler lookup; recover an untagged
      object only when runtime class and reviewed repository identify exactly one
      adapter; reject shared Wan classes without enough identity; and publish
      allowed modes from backend metadata. Fake pipelines only; assets: none.
    - [x] **P0.3b.4 Generic audio mode/task/input contract:** bind each ACE-Step
      and Stable Audio mode to its valid task, required/forbidden audio inputs,
      duration/interval rules, and backend-driven task visibility. Keep
      `extract`, `lego`, and `complete` unavailable until their missing dedicated
      inputs and modes are designed. Fake pipelines only; assets: none.
    - [x] **P0.3b.5 Generic client filtering:** remove repository-name/family
      inference from ModelSelect. Apply backend class/id filters and dynamic
      field mutations only; prove an opaque future family works without a
      production model-name branch and a stale tuple remains blocked. Mocked
      browser only; assets: none.
    - [x] **P0.3b.6 Generic auxiliary LoRA identity:** replace mutable and
      path-only `custom_lora` payloads with one versioned, model-neutral
      descriptor. Hub weights require an exact commit, SHA-256, managed-cache
      containment, and a literal lowercase `.safetensors` filename; local
      weights require an existing resolved Safetensors file and a content
      digest. Revalidate the descriptor immediately before every Modular,
      stack/mix, and hotswap load so a hand-authored graph cannot bypass the
      producer node. Keep scheduler overrides inside a reviewed Diffusers
      scheduler contract. Fake pipelines and temporary tiny Safetensors only;
      assets: none.
    - [x] **P0.3b.7 Paired checkpoint:** run focused and complete backend/client
      gates, mirror the reviewed client bundle, perform a fresh HTTP smoke, and
      record paired references. Live generation and Gallery assets are not
      required for this contract-only phase.
    - Status 2026-08-09: P0.3b.1 through P0.3b.6 are implemented and independently
      re-reviewed. Image, video, and audio loaders now bind exact class, mode,
      source, repository, revision, runtime identity, and backend-issued dynamic
      contracts. Unknown or ambiguous adapters fail closed. ModelSelect consumes
      only the backend class/id filter grammar, including opaque future-family
      tests, and contains no new family-name selection branch. Cache-ignored mode
      retags reuse resident pipelines while invalidating descendants so every
      action reruns its authoritative preflight. The generic auxiliary LoRA
      descriptor revalidates an exact commit, managed alias or contained local
      path, SHA-256, and a nonempty Safetensors header before every lifecycle
      mutation. Its optional scheduler override is restricted to a bounded,
      reviewed FlowMatch contract and is preconstructed for the same pipeline.
    - Evidence 2026-08-09: the independent combined P0.3b matrix passed (`436
      passed, 880 subtests, 2 platform skips`), and the final complete backend
      suite passed (`873 passed, 2 skipped, 1 warning, 1100 subtests`). Ruff
      `E9,F`, dependency validation (`78 packages compatible`), preflight
      (`ready: true`), and both diff checks passed. Client `npm run check` passed
      in 44.8 seconds, including format, lint, typecheck, contract suites,
      production build, and the bundle budget (`522401 / 523264` gzip bytes).
      Six focused backend-driven mocked-browser contracts passed, and a fresh
      real-browser smoke connected to the backend and materialized the generic
      Layered workflow at the backend-bound 640 by 640 resolution.
    - Checkpoint closure 2026-08-10: the implementation is recorded in backend
      `91c9a36` and client `28b12b7`; the exact-target follow-up checkpoints are
      backend `8fb2cb9` and client `c3e8a17`. Four Win32 shared-control snapshots
      were generated by the existing Playwright contract, reviewed for font,
      clipping, overlap, disabled/error-state, and portalled-listbox correctness,
      and committed as client `d226c4b`. The unmodified `npm run check:ui` then
      passed all 2 shared-control and 86 mocked Studio cases. The current client
      `npm run check` passed with a 523003-byte gzip bundle, and the current
      complete backend gate passed 1113 tests with 4 skips and 1688 subtests.
      The reviewed build matched all 26 mirrored files byte-for-byte; all 317
      backend-owned local Gallery files remained present; and a fresh backend
      returned HTTP 200 for the app, bundle, health, and runtime status before
      port 8088 was released. No model, media, Gallery asset, Transformers
      installation, network model download, or GPU execution occurred. The
      separate P0.5 clean-base migration remains required before the base
      installer can claim that Transformers and PEFT are absent by default.
  - [x] **P0.3c Upstream workflow and component truth closure**
    - Backend: remove false runnable claims first; validate every Modular
      action against an upstream workflow or reviewed fixed block sequence;
      remove Flux-family Guider ports, add Wan Guider ports, add the two guiders
      present at the pin, and record latest-only guiders behind the pin update.
      Add missing fields only where their complete state flow is understood.
      Correct the standard Flux adapter's modern `true_cfg_scale` and negative-
      prompt forwarding so the generic guidance field does not silently target
      the wrong upstream parameter.
    - Client: consume the corrected action set and never infer support from a
      family name, a generic port, an installed artifact, or a nearby workflow.
    - Tests: all eleven registered Modular classes, all non-null node actions,
      block input/output/component closure, guider compatibility, and explicit
      empty runnable sets. New modes remain `contract_only`, Expert-visible,
      and excluded from Auto.
    - [x] **P0.3c.1 Remove false claims and freeze the upstream matrix:** derive
      closure tests from the pinned workflow maps or reviewed fixed block
      sequences; withdraw currently advertised modes that cannot be assembled,
      including Modular SDXL inpaint, before adding any replacement. Record
      unsupported reasons in capabilities instead of exposing a broken action.
      Static/no-weight tests only; assets: none.
      - Status 2026-08-09: a data-only truth matrix now freezes the exact
        workflow maps or reviewed fixed top-level block sequences for all eleven
        registered Modular classes. Every advertised Modular mode resolves to a
        current generic action sequence, covers its upstream-required user
        inputs, and has explicit producer-to-consumer state edges. At that
        checkpoint, SDXL Modular inpaint was removed from runnable modes and its
        then-missing mask, masked-latent, crop, and overlay state was published
        as an additive schema-v2 `unsupportedModes` reason. The later P0.3c.3
        internal base-inpaint closure below supersedes that state-gap detail
        without promoting the mode. The existing Flux ControlNet reason is also
        normalized. Current clients safely ignore that new optional detail while
        continuing to treat the corrected `runnableModes` list as authoritative.
        Flux2 Klein retains its legacy Modular `modelType` identifier for
        compatibility, but now identifies only the standard
        `Flux2KleinPipeline`, generic image backend, and
        `flux2-klein:direct` execution profile; a missing referenced profile
        empties its public runnable set.
      - Evidence 2026-08-09: the focused matrix passed (`8 passed, 24
        subtests`); the adjacent capability, upstream Modular, and direct-profile
        batch passed (`46 passed, 158 subtests`). Repository-wide Ruff E9/F and
        `git diff --check` passed. The checks instantiated only no-weight block
        definitions and generated no model downloads, media, or assets.
    - [x] **P0.3c.2 Guider and component-port truth:** remove Guider ports from
      Flux, Flux Kontext, and Flux2 Klein; add the upstream Wan T2V/I2V Guider
      ports; expose the two guiders present at the pin; keep the latest-only
      guider behind a Diffusers-pin update. Validate every port against the
      instantiated no-weight block contract. Assets: none.
      - Status 2026-08-09: the eleven registered no-weight denoise block
        contracts now enforce exact generic component-port closure. Flux, Flux
        Kontext, and Flux2 Klein expose no Guider port; Wan T2V and I2V do.
        `AdaptiveProjectedMixGuidance` and `PerturbedAttentionGuidance` use the
        pinned constructor contracts, while latest-only
        `MagnitudeAwareGuidance` remains unregistered. Perturbed Attention maps
        the generic Layers payload to `perturbed_guidance_config` and rejects
        missing, malformed, or incompatible layer data before construction.
        Additional model-to-layer-stack discovery remains tracked under P0.3e;
        no unreviewed Wan stack name was guessed here.
      - Evidence 2026-08-09: focused upstream contracts passed with `34 passed,
        116 subtests`; the adjacent Modular/schema/identity batch passed with
        `113 passed, 1 warning, 185 subtests`; repository-wide Ruff E9/F and
        `git diff --check` passed. These were CPU/no-weight tests and generated
        no media, downloads, or assets.
      - Superseded by P1.3 on 2026-08-12: the class was verified at the existing
        pin under the official `diffusers.guiders` namespace even though the
        top-level lazy export omits it, and it is now registered without a pin
        change.
    - [x] **P0.3c.3 Complete generic state flows:** add mask/processed-mask/
      overlay state for SDXL and Qwen inpaint, a generic IP-Adapter encoding
      action for SDXL, and first/last-frame state for Wan I2V. Preserve the
      exact Qwen Layered controls recorded below. Add generic
      control-edit/control-inpaint fields only as
      complete backend adapters, never as family-named frontend nodes. Tiny
      fixture tests may run locally; assets: none.
      - Preparatory truth cleanup 2026-08-09: removed the stale
        `QwenImageEditPlusModularPipeline:inpaint` claim from backend legacy
        mode metadata and the client's offline profile and Auto fallbacks. The
        blocked mask-contract explanation remains available, schema-v2
        `runnableModes` remains authoritative, and the standard
        `QwenImageEditInpaintPipeline` inpaint/outpaint profile is unchanged.
        An imported stale Edit Plus inpaint form now receives the exact backend
        unsupported-mode blocker and cannot become Auto-ready. Focused backend
        tests passed (`4 passed, 39 subtests`), and the Auto execution guard
        passed (`1 passed, 3 subtests`); client template/contract tests passed
        (`57 passed`), with client format, lint, and typecheck plus
        backend Ruff E9/F and both diff checks passing. Static tests only; no
        downloads, model execution, generated media, or assets. This does not
        complete the P0.3c.3 state-flow work.
      - Preparatory Layered-control closure 2026-08-09: the backend's existing
        generic Encode Prompt and Encode Image actions now publish the targeted
        pinned Layered controls: a shared `resolution` constrained to 640 or
        1024 with default 640, text-only `use_en_prompt` defaulting false, and a
        positive `max_sequence_length` capped at 1024 with default 1024. The
        client removed its Layered class-name branch and consumes a small
        backend-issued `studioBinding` grammar through the existing managed
        graph. Opaque field names synchronize only when their complete binding
        signatures match. Unknown, extra, malformed, direct-dimension identity,
        and out-of-envelope dimension metadata fail closed; valid numeric node
        edits canonicalize the source and every peer to the target type. This
        slice does not change Auto modes or the existing Layered template; the
        new control evidence is contract-only.
      - Evidence 2026-08-09: backend upstream/schema/fake-pipeline tests passed
        (`49 passed, 134 subtests`), and the adjacent Modular suite passed (`74
        passed, 158 subtests`); client generic graph contracts passed (`38
        passed`), the focused mocked-browser dynamic-refresh test passed (`1
        passed`), and client format, lint, and typecheck plus backend Ruff E9/F
        and both scoped diff checks passed. These checks loaded no weights and
        generated no downloads, media, templates, or assets.
      - Preparatory seed-state closure 2026-08-09: every one of the ten
        registered VAE encoder actions whose pinned upstream block exposes
        `generator` now declares the same bounded generic seed field. Encode
        Image validates the scalar before pipeline initialization, constructs a
        Torch generator on the managed pipeline execution device, and forwards
        no raw seed. Caller-supplied generator objects and backend/upstream
        generator-contract disagreement fail before model identity resolution or
        block initialization. Wan I2V's distinct image encoder truthfully remains
        unchanged because it has no upstream generator input. Focused
        schema/upstream/fake-pipeline tests passed (`54 passed, 160 subtests`),
        the adjacent Modular/graph/custom-identity batch passed (`137 passed, 1
        warning, 309 subtests`), and the independent root matrix passed (`121
        passed, 1 warning, 229 subtests`). Ruff E9/F and diff checks passed. No
        dependency, client, graph, template, asset, model, GPU, inference, or
        Transformers-install action was involved.
      - Preparatory opaque-route closure 2026-08-09: built-in Qwen Image and
        Qwen Image Edit VAE encoders now carry post-sampling generator state,
        processed mask, overlay state, and exact latent pairing through generic
        Encode Image -> Denoise -> Decode Latents route handles. Qwen Image Edit
        Plus uses the same route only for generator continuity and rejects
        inpaint state. Each process-local route is bound to one successful
        Models Loader execution, exact component role and model-id inventory,
        stage, seed, and the exact paired Torch tensor through a weak identity
        reference. Edit Plus multi-image state instead seals a bounded ordered
        list of exact tensor identities. Serialized, nested, stale,
        cross-loader, role-swapped, same-loader cross-paired, reordered, and
        dead-reference values fail before pipeline initialization. Route-aware
        node-cache comparison preserves only exact tensor identities, so an
        equal-valued clone cannot bypass that validation. Denoise retries clone
        the stored post-VAE generator state, and every supported Qwen Decode
        requires the matching Denoise route.
      - The client selects this topology only from generic live handles: every
        Modular graph with a complete route contract connects Denoise to Decode;
        image flows additionally connect Encode Image to Denoise; and native
        inpaint replaces the Apply Mask fallback only after the mask plus both
        route chains are complete and have exact input/output displays plus one
        identical unpadded opaque type. Partial, malformed, mismatched, or
        out-of-order node definitions keep finalization and Run pending. The
        schema-v2 finalization proof binds the sorted exact endpoint set and is
        revalidated against live managed edges, so preserved-ID endpoint edits
        and forged checksums cannot restore readiness. No model or repository
        name controls this branch.
      - Preparatory standalone-component provenance 2026-08-09: generic
        AutoModelLoader outputs now carry an immutable, process-local binding
        that seals the component role, ComponentsManager identity, canonical
        source/repository/revision/subfolder/class tuple, and config SHA-256.
        Per-loader current-publication identity revokes resident A after A->B;
        resident reuse and node-cache hits require the exact live binding, while
        config TOCTOU, relabeling, metadata tampering, and stale publications
        fail before class resolution or component initialization. This is the
        dormant provenance foundation for the next Qwen ControlNet route slice;
        it does not add ControlNet fields, runtime behavior, modes, templates,
        or client branches. Cache HTTP responses now also reject connector and
        other opaque static types with a controlled `400`; only declared media
        and text families are served.
      - Evidence 2026-08-09: the final dedicated route/provenance suite passed
        (`49 passed, 61 subtests`), cache HTTP security passed (`20 passed, 29
        subtests`), and the complete backend gate passed (`931 passed, 2
        skipped, 1 warning, 1211 subtests`). Repository-wide Ruff E9/F,
        dependency validation, preflight, and diff checks passed. Client graph
        contracts passed (`39 passed`), proof/template contracts passed (`58
        passed`), the focused out-of-order mocked-browser transition passed (`1
        passed`), and full `npm run check` passed with the unchanged bundle
        budget (`523126 / 523264` gzip bytes). The reviewed bundle was mirrored
        byte-for-byte and fresh HTTP smoke checks returned `200` for `/`,
        `/assets/index.js`, and `/health`. These were CPU/static, no-weight, and
        mocked-browser checks; they made no model downloads, live inference,
        generated media, template, capability, optional-runtime, or Gallery-
        asset changes.
      - Preparatory Qwen ControlNet route closure 2026-08-09: the generic
        ControlNet action now exposes a bounded seed plus opaque route input and
        output only when the pinned upstream ControlNet VAE block exposes its
        generator. Text control starts from that seed; an image route continues
        the exact post-main-VAE generator snapshot. The output seals the exact
        current standalone ControlNet publication and an ordered weak identity
        for the resulting control latent tensor or bounded tensor list. Denoise
        requires the matching ControlNet bundle, component publication, seed,
        and inherited image state, continues the generator once, and emits the
        already-reviewed Decode route. No large image latent is copied into the
        opaque carrier or routed through ControlNet.
      - Cache candidates re-resolve and validate the complete effective
        image/control inputs, reserved fields, component roles, and route state;
        same-object nested mutation cannot reuse a cached result. ControlNet
        revalidates both the VAE and standalone publications after block
        initialization, after component updates, immediately before the
        upstream call, and again after it before publishing any bundle or route.
        Init-time, call-time, stale-publication, cross-loader, cloned-latent,
        reordered-list, nested-reserved-input, and retry paths all fail closed
        at their reviewed boundary. The SDXL bundle-only branch remains
        unchanged and has an exact legacy-output regression.
      - The client adopts this route using only generic live handles, display
        directions, and one exact opaque type. Legacy ControlNet graphs retain
        their bundle-only topology. A complete contract connects ControlNet to
        Denoise to Decode, and adds Encode Image to ControlNet only when that
        prefix is published; ordinary image/control latent and bundle edges stay
        typed and direct. Partial, mismatched, or out-of-order definitions keep
        the last stable topology and Run proof pending. Seed synchronization is
        field-driven across Encode Image, ControlNet, and Denoise; no model or
        repository name selects the client branch.
      - The existing Qwen ControlNet component is now pinned consistently in
        the executable catalog, backend capability requirements, saved graph,
        workflow manifest, and client requirement to official repository
        `InstantX/Qwen-Image-ControlNet-Union` commit
        `b13036f066d6dee7c20513e263d3d673055e9de8`. Metadata-only review confirmed
        the Apache-2.0 repository, configuration, and Safetensors files; no
        weight was downloaded. This slice promotes no new mode and generates no
        template or Gallery asset.
      - Evidence 2026-08-09: the final focused backend route/upstream/truth/schema
        matrix passed (`120 passed, 247 subtests`), the adjacent matrix passed
        (`134 passed, 1 warning, 81 subtests`), the exact adversarial boundary
        matrix passed, and the complete backend gate passed (`954 passed, 2
        skipped, 1 warning, 1242 subtests`). Repository-wide Ruff E9/F,
        dependency validation, preflight, and diff checks passed. Client graph
        contracts passed (`39 passed`), the focused late-definition browser test
        passed (`1 passed`), and full `npm run check` passed with the unchanged
        budget (`523065 / 523264` gzip bytes). The validated client bundle was
        mirrored byte-for-byte; fresh HTTP smoke checks returned `200` for `/`,
        `/assets/index.js`, and `/health`. These were CPU/static, no-weight, and
        mocked-browser checks with no model download, inference, generated
        media, optional-runtime installation, or new asset.
      - Preparatory combined-Qwen state-flow closure 2026-08-10: a nonadvertised
        pinned truth table now records the exact upstream block order, required
        inputs, generic action order, and ordered typed/opaque state edges for
        Qwen img2img, inpaint, ControlNet img2img, and ControlNet inpaint. A
        no-weight fake-action matrix executes all four paths through the real
        Encode Image, optional ControlNet, Denoise, and Decode adapters while
        proving generator continuation, mask/overlay preservation, and exact
        source/control separation. Public capabilities, profiles, modes,
        templates, Auto candidates, and assets are unchanged.
      - The generic client may preserve that dormant combined topology only for
        canonical `edit_image` or `inpaint` bindings that already own a complete
        distinct control-image loader, ControlNet model loader, and ControlNet
        action. Legacy text-control retains its single-loader fallback. Missing,
        partial, detached, duplicated, wrong-mode, or stale-form role groups
        remain pending and mint no proof. Binding mode/model/fingerprint now
        gate value application, edge creation, restore, waiting, and final proof
        publication; every participating optional node and exact endpoint is
        sealed by the binding-scoped schema-v2 proof. No model or repository
        name selects this topology.
      - Evidence 2026-08-10: focused backend truth/route tests passed (`80
        passed, 128 subtests`), and the complete backend gate passed (`957
        passed, 2 skipped, 1 warning, 1254 subtests`). Client graph contracts
        passed (`39 passed`), independent adversarial restore/finalizer review
        found no remaining blocker, and full `npm run check` passed with the
        unchanged bundle budget (`523078 / 523264` gzip bytes). Repository-wide
        Ruff E9/F, dependency validation, preflight, and both diff checks passed.
        The validated bundle was mirrored byte-for-byte (SHA-256
        `5beab30710b35c50d105f091cb3d578079b0138fc20fbacc20cce5d818208baf`);
        fresh HTTP smoke checks returned `200` for `/`, `/assets/index.js`, and
        `/health`, and port 8088 was free after the worker stopped. These were
        CPU/static and fake-action checks with no model download, inference,
        generated media, optional-runtime installation, template, or asset
        change.
      - Preparatory SDXL base-inpaint state-flow closure 2026-08-10: a
        nonadvertised pinned truth entry now records the exact upstream block
        sequence, required inputs, generic action order, and typed/opaque state
        edges for base inpaint. The no-weight generic action route preserves
        post-VAE generator state, typed mask and masked-image latents, crop
        overlay state, exact tensor identity and mutation version, and the
        exact resident VAE identity plus derived geometry across cache,
        initialization, component update, call, and output boundaries. Bounded
        tensor and PIL crop validation fails before expensive execution.
      - The generic client latches this dormant typed topology only after a live
        Denoise VAE handle or persisted managed VAE edge is observed, then
        requires the exact matching VAE, mask, masked-latent, and route handles
        before finalization. Missing or malformed later definitions remain
        pending instead of downgrading to Apply Mask or a route-only topology;
        the latch is binding-fingerprint-scoped, so a new route-only Qwen
        binding does not inherit it. The complete native topology has an exact
        16-edge schema-v2 proof, and no model or repository name selects it.
      - Public capabilities, profiles, modes, Auto candidates, templates, and
        Gallery assets are unchanged. SDXL ControlNet inpaint, ControlNet
        Union, IP-Adapter, and combined variants remain explicitly outside this
        base contract, and no live model qualification was performed.
      - Evidence 2026-08-10: the independent focused backend matrix passed
        (`152 passed, 298 subtests`), its exact adversarial slice passed (`14
        passed, 39 subtests`), and the complete backend gate passed (`969
        passed, 2 skipped, 1 warning, 1296 subtests`). Client graph contracts
        passed (`39 passed`), independent review and full `npm run check`
        passed with the unchanged budget (`523030 / 523264` gzip bytes).
        Repository-wide Ruff E9/F, dependency validation, preflight, and both
        diff checks passed. The validated client bundle was mirrored
        byte-for-byte (1,298,272 bytes; SHA-256
        `e2811ebfc6c9d5cd78f294216daaa69667843af7ae57a1ade0144e1d3eda1ba8`);
        fresh HTTP smoke checks returned `200` for `/`, `/assets/index.js`, and
        `/health`, and port 8088 was free after the process tree stopped. These
        were CPU/static and fake-action checks with no model download,
        inference, generated media, optional-runtime installation, template,
        or Gallery-asset change.
      - Preparatory Wan split-route closure 2026-08-10: the existing
        backend-declared Modular `image_to_video` action now preserves exact
        image embeddings, condition latents, two-pass area-budget geometry,
        post-VAE generator state, source-media snapshots, resident
        component/processor identity, and bounded effective configuration
        through Image Embeddings, Encode Image, Denoise, Decode, cache, retry,
        and TOCTOU boundaries. Raw
        mode-specific frame latents remain internal to the VAE action.
        First/last-frame truth and typed topology are structural only. The
        official FLF repository has a distinct CLIP preprocessing and
        transformer positional-embedding contract and is not an executable
        catalog artifact, so `last_image` is rejected before contract lookup,
        component resolution, block initialization, or cache reuse. No mode,
        profile, template, asset, or dependency pin was added.
      - The generic client recognizes this dormant Modular video topology only
        from complete live roles, handles, display directions, and one exact
        opaque route type. An already-present I2V group has an exact 17-edge
        schema-v2 proof; the preparatory FLF group adds a distinct last-image
        loader and two last-image edges for an exact 19-edge proof. Partial,
        removed, remapped, duplicated, type-mismatched, or retained hidden role
        groups remain pending and cannot downgrade to the standard video
        facade. No model/repository name selects the branch, and no selectable
        Studio profile or template was added.
      - Evidence 2026-08-10: the dedicated CPU/no-weight Wan boundary suite
        passed (`30 passed, 81 subtests`); the route/schema/truth/upstream matrix
        passed (`165 passed, 382 subtests`); and the adjacent action recovery,
        output, offload, and custom-identity matrix passed (`124 passed, 1
        warning, 106 subtests`). Two independent frozen audits reproduced the
        exact cataloged I2V processor/component contracts and adversarial
        device, tensor, generator, crop, cache, and TOCTOU boundaries without a
        model download. The complete backend gate passed (`999 passed, 2
        skipped, 1 warning, 1377 subtests`); repository-wide Ruff E9/F,
        dependency validation, preflight, and diff checks passed.
      - Client graph contracts passed (`40 passed`) and full `npm run check`
        passed with the unchanged budget (`523110 / 523264` gzip bytes). The
        reviewed build was mirrored exactly as `index.js` (1,006,250 bytes,
        SHA-256 `801f0f5d06c2d073b77deee696ce048039ab5f68fa6b30da7c09480d9d7459f3`)
        plus its static `studio-templates.js` chunk (296,450 bytes, SHA-256
        `e3c722e395f5f93751d14b6e936d5d07d85e5a65e3b73c428d8f26f74aaaddea`).
        Fresh HTTP smoke checks returned `200` for `/`, `/assets/index.js`,
        `/assets/studio-templates.js`, and `/health`, and port 8088 was free
        after the process tree stopped. These were CPU/static, no-weight, and
        fake-action checks with no live inference or model, media, template, or
        Gallery-asset download/generation.
      - Preparatory SDXL ordinary-ControlNet composition closure 2026-08-12:
        pinned truth now records the exact upstream
        `controlnet_image2image` and `controlnet_inpainting` block order,
        required inputs, generic action sequence, and typed/opaque graph edges.
        The existing VAE route may reach Denoise alongside the generic
        ControlNet bundle only when the standalone loader publication is the
        exact ordinary `ControlNetModel` class. Denoise resolves its exact
        managed component before pipeline initialization, proves that the same
        object was installed, and rechecks publication plus resident identity
        at cache, pre-call, and post-call boundaries. Missing bundles,
        prepared Qwen control latents, stale publications, Union components,
        and init/call-time resident swaps fail closed before a runnable output
        route is published. The upstream component remains installed on the
        Modular pipeline; only its ordinary control image and scale inputs are
        forwarded to the call.
      - This slice is internal and no-weight. It does not advertise a new
        mode, choose or download an SDXL ControlNet artifact, add an execution
        profile, Auto candidate, template, or Gallery asset, or qualify model
        output. ControlNet Union and IP-Adapter remain rejected as distinct
        component/weight contracts. Focused truth, route, upstream, and Wan
        regression tests passed (`173 passed, 456 subtests`). The complete
        backend gate passed (`1213 passed, 4 skipped, 1 existing warning, 2048
        subtests`); Ruff E9/F, `py_compile`, dependency validation (78
        compatible packages), preflight, and diff checks passed. The unchanged
        compatible client passed complete `npm run check`, including its graph
        contracts and a 523108/523264-byte gzip bundle (28 bytes below the
        stricter 523136-byte target).
      - Public mode/profile promotion, templates/assets, and live qualification
        for the completed Qwen and SDXL base/ordinary-ControlNet flows remain
        outstanding; SDXL ControlNet Union and IP-Adapter combinations remain
        unfinished, so P0.3c.3 stays incomplete.
      - Preparatory SDXL ControlNet Union composition closure 2026-08-12: the
        same generic ControlNet action now declares an ordinary/Union selector.
        The existing client field-action grammar hides the Union control-type
        index for ordinary execution and reveals it only for Union. The index is
        a bounded generic integer rather than an artifact-specific semantic
        name. The action requires the selected variant to match the exact
        process-local `ControlNetModel` or `ControlNetUnionModel` publication;
        Denoise additionally validates the index against the exact resident
        Union model's bounded `num_control_type`, installs that same object, and
        repeats the publication/resident checks at cache, initialization,
        pre-call, and post-call boundaries. Pinned truth records the complete
        single-control Union image-to-image and inpaint block/action/edge flows.
      - This Union slice remains internal, CPU/no-weight, and single-control. It
        adds no public mode, execution profile, selected repository, Auto
        candidate, template, Gallery asset, model download, or output
        qualification. Multi-ControlNet, IP-Adapter, public promotion, and live
        qualification remain open, so P0.3c.3 stays incomplete.
        Focused truth/route/upstream/Wan regressions passed (`173 passed, 459
        subtests`); the complete backend gate passed (`1213 passed, 4 skipped,
        1 existing warning, 2051 subtests`). Ruff E9/F, `py_compile`, dependency
        validation (78 compatible packages), preflight, and diff checks passed.
        The compatible client passed complete `npm run check`, including 37
        run/field-action tests and the unchanged 523108/523264-byte gzip bundle.
      - Preparatory SDXL single-IP-Adapter state-flow closure 2026-08-12: a new
        generic **IP-Adapter Embeddings** action derives its fields from the
        selected registered pipeline contract and matches the pinned upstream
        `StableDiffusionXLIPAdapterStep`. All nine upstream single-adapter
        text/image/inpaint plus ordinary/Union ControlNet compositions now have
        exact nonadvertised block, action, and typed-state edge truth. The only
        admitted descriptor is the Apache-2.0 `h94/IP-Adapter` repository at
        exact commit `018e402774aeeddd60609b4ecdb7e298259dc729`, standard SDXL
        weight `sdxl_models/ip-adapter_sdxl.safetensors`, reviewed size
        702,585,376 bytes, and SHA-256
        `ba1002529e783604c5f326d49f0122025392d1d20ac8d573b3eeb3e6dea4ebb6`.
        Execution resolves and hashes that file from the local Hub cache only;
        it never installs or downloads. The pinned image encoder is likewise
        local-only and must retain its exact CLIP ViT-H geometry and processor
        contract.
      - The adapter action mutates only the exact resident SDXL UNet, supports
        one standard projection and one bounded scale, and issues a
        nonserializable process-local receipt sealing loader/UNet/Guider,
        adapter parameter versions, encoder/processor, source pixels, and both
        embedding tensor identities. Denoise validates the receipt before
        initialization, after component installation, before the upstream call,
        after the call, and on cache reuse. A missing bundle, copied state,
        wrong loader/component/Guider, changed image/tensor/parameter/scale, or
        resident swap fails closed. Models Loader removes only a current owned
        adapter before minting a replacement loader receipt and rejects
        unreceipted mutation state.
      - This is an internal/manual contract-only slice. It adds no public mode,
        execution profile, Auto candidate, template, Gallery asset, installer,
        dependency cutover, or live output claim. The optional Transformers
        runtime must already have been explicitly installed and verified;
        registry discovery and graph execution do not install it. Multiple
        adapters, Multi-ControlNet, public promotion, templates/assets, and live
        qualification remain open, so P0.3c.3 stays incomplete.
      - Evidence 2026-08-12: the focused IP-Adapter, route-state, workflow-truth,
        upstream-contract, artifact-catalog, and local-resolution matrix passed
        `162 tests` with `411 subtests`; the complete backend gate passed
        `1228 tests` with `4 skips`, `2086 subtests`, and only the existing
        upstream Diffusers `torch_dtype` deprecation warning. Ruff E9/F,
        `py_compile`, dependency validation (78 compatible packages), preflight,
        port, and diff checks passed. The unchanged compatible client passed
        complete `npm run check`, including the `523108/523264`-byte gzip
        bundle. Its exact mocked-browser generic `node_definition` contract
        passed `1/1` and proved that backend-selected node/field metadata updates
        without replacing current field values. These are CPU/static/unit/
        contract/mocked-browser results: no Hub download, adapter installation,
        model execution, generated media, or live output qualification occurred.
      - P0.3c contract-only closure 2026-08-12: the completed base inpaint flow
        is now an exact high-level `inpaint` mode on the existing generic SDXL
        Modular capability. That whole capability is explicitly
        `contract_only`, pinned to
        `stabilityai/stable-diffusion-xl-base-1.0@462165984030d82259a11f4367a4eed129e94a7b`,
        and publishes exact reference-image plus mask input requirements. It has
        no execution profile and sets Auto, template, and Gallery eligibility
        false. The combined ordinary/Union ControlNet and single-IP-Adapter
        routes remain constructible manual generic-node compositions; no new
        model-named mode or client branch was added. Multi-ControlNet and
        multiple-adapter execution are outside the pinned 18-workflow upstream
        matrix and require their own future artifact/state-flow review rather
        than keeping this scoped truth phase open.
      - Evidence 2026-08-12: the final P0.3c truth/capability/route/optional-
        runtime matrix passed `227 tests` with `1 skip` and `729 subtests`; the
        complete backend gate passed `1228 tests` with `4 skips`, `2088
        subtests`, and only the existing upstream Diffusers `torch_dtype`
        deprecation warning. Ruff E9/F, dependency validation (78 compatible
        packages), preflight/port, and diff checks passed. The unchanged client
        passed complete `npm run check` and its `523108/523264`-byte gzip
        budget. A fresh HTTP smoke reported ready and returned schema-v2 SDXL
        Modular contract-only truth with exactly four high-level modes, the
        pinned revision, zero execution profiles, and all three eligibility
        flags false; the owned server tree was stopped and port 8088 was free.
        This is CPU/static/unit/contract/HTTP evidence only, not model or media
        execution or live qualification.
      - Wan FLF artifact admission 2026-08-12: the Apache-2.0 official
        `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers` checkpoint is pinned at exact
        commit `17c30769b1e0b5dcaa1799b117bf20a9c31f59d7`. Its 50-file snapshot
        contains 21 Safetensors files and no Python source. The existing generic
        `WanImage2VideoModularPipeline` Models Loader accepts it only as an exact
        reviewed repository variant and still requires the catalog revision.
        The loader index boundary recognizes only its exact concrete
        `CLIPProcessor`/`CLIPVisionModelWithProjection` declarations; the
        Image Embeddings action loads the pinned block's exact
        `CLIPImageProcessor` view locally. Image Embeddings and Encode Image bind
        `image2video` to the original I2V artifact and `flf2v` to the FLF
        artifact before block initialization and cache reuse. Swapping only
        `last_image`, repository, revision, or component type fails closed. The
        generic Model Manager install route now resolves an omitted revision to
        the catalog commit for every reviewed repository, so the existing
        model-select Install control cannot fetch mutable `main` for this or
        another curated artifact; uncataloged user-selected repositories retain
        their prior behavior. The affected loader/route/catalog/download matrix
        passed 296 tests with 747 subtests. The complete backend gate passed
        1,211 tests with 4 skips, the existing Diffusers deprecation warning,
        and 2,041 subtests; Ruff E9/F, `uv pip check` (78 packages), preflight,
        `py_compile`, JSON parsing, and diff checks passed. A no-weight exact
        config probe read the pinned Hub `model_index.json` and image-processor
        config, admitted their exact reviewed types, normalized only the
        validated FLF image-processor load contract to `CLIPImageProcessor`,
        and observed its 224-pixel shortest-edge/center-crop configuration. The
        temporary config-only snapshot was removed. This is metadata,
        no-weight, unit, contract, and fake-action evidence only; no FLF model
        weights, inference, generated video, public template, or Gallery asset
        were downloaded or exercised. The unchanged compatible client passed
        complete `npm run check` with a 523108/523264-byte gzip bundle, 28 bytes
        below the stricter 523136-byte target; its mocked Model Manager install
        flow passed 1/1 in 5.2 seconds.
        Public FLF mode/profile/template promotion, assets, and live execution
        remain outside this slice.
    - [x] **P0.3c.4 Standard signature and adapter truth:** correct Flux
      `true_cfg_scale`/negative-prompt forwarding and register missing standard
      image classes only when each maps to an existing generic action with an
      exact fake-pipeline signature test. Keep KV/full Flux2 and other
      artifact-sensitive variants deferred until their runtime artifacts are
      reviewed. Assets: none.
      - Status 2026-08-10: modern Flux text, img2img, inpaint, and Kontext
        adapters now bind the generic guidance value to `true_cfg_scale` and
        forward the negative prompt only through an upstream signature that
        declares it. Eleven pinned standard classes now reuse the existing
        generic Generate/Edit/Inpaint actions: SDXL base/img2img/inpaint; Qwen
        Image img2img/inpaint and Edit/Edit Plus; Z-Image img2img/inpaint; Flux
        Kontext inpaint; and Flux2 Klein inpaint. Every class uses an already
        reviewed immutable base artifact and remains contract-only with no Auto
        profile or public template. Full Flux2, Flux2 KV, ControlNet/composite
        variants, Qwen Layered, Z-Image Omni, and SDXL instruct-pix2pix remain
        unregistered until their extra inputs or artifacts are reviewed.
      - Evidence 2026-08-10: the focused image registry passed 73 tests with 2
        skips and 177 subtests. The adjacent profile, capability, artifact,
        offload, Qwen-inpaint, and optional-runtime matrix passed 99 tests and
        176 subtests. The complete backend gate passed 1116 tests with 4 skips,
        1740 subtests, and only the existing upstream Diffusers deprecation
        warning. Ruff E9/F, `py_compile`, dependency validation (78 compatible
        packages), preflight, and diff checks passed. Tests inspected the pinned
        upstream signatures and executed every new adapter through fake
        pipelines; no model, artifact, media, network, or GPU execution occurred.
    - [x] **P0.3c.5 Contract-only exposure:** publish newly complete modes and
      already implemented but unprofiled video/audio adapters as
      `contract_only`, with backend-driven parameters and no Auto eligibility.
      Add templates only after graph-contract validation; generation and public
      Gallery media run later on the qualification machine.
      - Status 2026-08-11: `/model_capabilities` now publishes every registered
        but unprofiled generic Diffusers adapter as an exact experimental
        `contract_only` record: 13 standard image adapters, five video adapters,
        and Stable Audio. Each record names one generic loader, exact pipeline
        class/mode set, reviewed repository and immutable catalog revision,
        backend parameter aliases, and mode input contract. Contract-only
        records explicitly disable Auto, template, and Gallery eligibility;
        they have no execution profile or optional-runtime execution
        requirement and remain outside the primary supported capability list.
      - Evidence 2026-08-11: the adapter/profile/upstream-truth matrix passed
        267 tests with 2 skips and 777 subtests. The complete backend gate
        passed 1117 tests with 4 skips and 1759 subtests, with only the existing
        upstream Diffusers deprecation warning. Ruff E9/F, dependency
        validation (78 compatible packages), preflight, `py_compile`, and diff
        checks passed. Registry closure tests prove the published set is
        exactly the image/video/audio adapter set minus profiled classes and
        that every artifact is immutably cataloged. No model, artifact, media,
        network, GPU execution, template, or Gallery asset was used or changed.
    - [x] **P0.3c.6 Paired checkpoint:** run focused and complete backend/client
      gates, mirror the reviewed client bundle, perform a fresh HTTP smoke, and
      update the support matrix. No large-model or media qualification is part
      of this phase.
      - Evidence 2026-08-11: paired backend `896661a8dc13` and client
        `d226c4bbc2e0` passed the checkpoint. The backend adapter/profile truth
        matrix passed 267 tests with 2 skips and 777 subtests; the full backend
        gate passed 1117 tests with 4 skips and 1759 subtests plus the existing
        upstream deprecation warning. Ruff E9/F, dependency validation,
        preflight, and diff checks passed. The unchanged client passed full
        `npm run check`, 2/2 shared-control browser tests, and all 86 mocked
        Studio browser tests. Its production bundle remained 523003/523264
        gzip bytes, 133 bytes inside the stricter 523136-byte safety target.
      - The 26-file client build was verified byte-for-byte against `web/` with
        no extra generated deployment files; all 317 preserved local Gallery
        files remained untouched. Exact deployed hashes were
        `index.js`=`4eb4230f9e3779c89a49a7155ffc56984e47d0c492d604780cafad3ac3e5e133`,
        `studio-templates.js`=`e8552942199e04fe3980da5b91550f7e9964af7894bc43e781d2bceaa62554d4`,
        and `graph-vendor.js`=`76e8c330da63ee5806ef230e2567ed978fca7005efd1cf679b8c5d99e8bfd337`.
        A fresh supervised HTTP smoke returned `200` for `/` and
        `/assets/index.js`, `/health` reported ready, and schema-v2
        `/model_capabilities` returned 20 supported, 25 experimental, and 19
        contract-only records with zero Auto-eligible or profiled contract-only
        entries. The owned process tree was stopped and port 8088 was free.
        No model download, inference, GPU workload, generated media, template,
        or Gallery publication occurred.
  - [x] **P0.3d Backend-owned Studio execution specifications**
    - Backend: make one exact-pair registry the source for execution profiles,
      capabilities, Auto requirements, loader choices, workflow validation,
      generic graph roles/edges, form bindings, ordered dynamic actions, schema
      version, and content hash. Validate every reference against `/nodes`.
    - Client: parse and fail-close the new schema. First materialize Flux
      Schnell and Flux Dev text-to-image from the same generic recipe so the two
      graphs differ only through backend data. Keep legacy schema-v2 handling
      during migration, but never fall back from a malformed new specification.
    - Tests: unknown nodes/parameters/handles, dangling edges, invalid bindings,
      stable hashing, model switch with identical topology, runtime hints and
      proof receipts bound to specification identity, and Auto candidate
      overrides restricted to declared bindings.
    - Evidence 2026-08-11: backend commit `fd258d8` owns the versioned Flux
      Schnell and Flux Dev text-to-image specifications, validates them against
      the live node registry, publishes their exact content hashes, and rejects
      mismatched execution receipts before graph execution. Client commit
      `642ea9c` strictly parses those specifications, materializes both models
      through one generic graph recipe, preserves node IDs/topology across the
      model switch, and seals the selected specification into finalization and
      runtime receipts. The exact backend tree passed 1,121 tests with 4 skips
      and 1,762 subtests; the focused specification matrix passed 104 tests with
      466 subtests. Ruff E9/F, `uv pip check` (78 compatible packages), preflight,
      and diff checks passed. The client specification contracts passed 136/136,
      the complete `npm run check` passed, and the complete mocked Studio browser
      suite passed 87/87. The production bundle was 522606/523264 gzip bytes,
      530 bytes below the stricter 523136-byte safety target. The mirrored build
      matched all 26 generated files byte-for-byte while preserving 317 backend-
      owned Gallery files. A fresh HTTP smoke served the exact entry bundle,
      reported 131 nodes, and published both specification IDs with hashes
      `studio-spec-v1-9cd1abb5` and `studio-spec-v1-d5ee399d`; its owned process
      tree was stopped and port 8088 was free afterward. This is static, unit,
      contract, mocked-browser, and local HTTP evidence only: no model download,
      model execution, generated media, or live workload qualification occurred.
  - [x] **P0.3e Remove remaining frontend and node model switches**
    - Migrate one exact pair per backend/client commit from `modelProfiles` and
      `graphBridge` into validated specifications. Then move loader component
      requirements, Denoise field visibility, Layers allowlists, Guider and
      Scheduler compatibility, readiness, and resource metadata into reviewed
      declarative overlays.
    - Composite creative workflows remain versioned template/recipe data, but
      reuse the same generic nodes and bindings. Imported/manual Expert graphs
      remain editable and do not silently become managed Studio graphs.
    - Tests: graph equivalence for every migrated pair, model switching,
      dynamic parameter refresh, input/output normalization, readiness, Auto
      fail-closed behavior, and removal of the corresponding class-name branch.
    - [x] `FluxKreaPipeline:text_to_image`: backend commit `96f70cb` moves its
      execution profile, capability, Auto resource contract, generic roles,
      topology, field bindings, and receipt hash into the specification
      registry. Client commit `80ac243` removes Krea from the Flux-family graph
      switch and resolves the direct loader and `FluxPipeline` class from the
      exact backend specification/profile. Schnell, Dev, and Krea retain the
      same managed node IDs and edge shape while their artifacts and receipt
      identities remain distinct. The focused backend matrix passed 66 tests
      with 321 subtests; the exact full backend tree passed 1,121 tests with 4
      skips and 1,762 subtests. The client execution-spec matrix passed 136/136,
      the complete `npm run check` passed, and the complete mocked Studio browser
      suite passed 87/87. The production bundle was 522633/523264 gzip bytes,
      503 bytes below the stricter 523136-byte safety target. The mirror matched
      all 26 generated files byte-for-byte and preserved 317 backend-owned
      Gallery files. Ruff E9/F, package compatibility, preflight, formatting,
      lint, type, and diff checks passed. This is static, unit, contract, mocked-
      browser, build, and local preflight evidence only; no Krea download, model
      execution, generated media, or live workload qualification occurred.
    - [x] `FluxDepthPipeline:control_image`: backend commit `a299d1d` moves its
      execution profile, capability, Auto resource contract, generic image
      loader, control-image loader, control generator, preview route, field
      bindings, and receipt hash `studio-spec-v1-2d8b881e` into the specification
      registry. Client commit `2de0c68` removes Depth from the Flux-family graph
      switch and resolves the direct facade plus `FluxControlPipeline` identity
      from the exact backend specification/profile. The focused backend matrix
      passed 67 tests with 321 subtests; the exact mirrored backend tree passed
      1,122 tests with 4 skips and 1,762 subtests. The client execution-spec
      matrix passed 136/136, the complete `npm run check` passed, and the full
      mocked Studio browser suite passed 87/87. The production bundle was
      522635/523264 gzip bytes, 501 bytes below the stricter 523136-byte safety
      target. The mirror matched all 26 generated files byte-for-byte while
      preserving 317 backend-owned Gallery files, and the local HTTP smoke
      returned 200 for the index and all seven referenced assets. Ruff E9/F,
      package compatibility, preflight, formatting, lint, type, and diff checks
      passed. This is static, unit, contract, mocked-browser, build, and local
      HTTP evidence only; no Depth download, model execution, generated media,
      or live workload qualification occurred.
    - [x] `FluxCannyPipeline:control_image`: backend commit `14fef9f` moves its
      execution profile, capability, Auto resource contract, reviewed compatible
      repair source, shared generic control-image recipe, field bindings, and
      receipt hash `studio-spec-v1-82045f56` into the specification registry.
      Client commit `784e3c7` removes Canny from the Flux-family graph and
      `FluxControlPipeline` class switches; Depth and Canny now reuse the exact
      same managed role IDs and edge shape while retaining distinct artifacts,
      profiles, and receipts. The focused backend matrix passed 95 tests with
      443 subtests; the exact mirrored backend tree passed 1,122 tests with 4
      skips and 1,762 subtests. The client execution-spec matrix passed 136/136,
      the complete `npm run check` passed, and the final full mocked Studio
      browser suite passed 87/87 after its legacy empty-capability fixture was
      corrected to serve the new exact schema-v2 Canny contract. The production
      bundle was 522626/523264 gzip bytes, 510 bytes below the stricter
      523136-byte safety target. The mirror matched all 26 generated files
      byte-for-byte while preserving 317 backend-owned Gallery files, and the
      local HTTP smoke returned 200 for the index and all seven referenced
      assets. Ruff E9/F, package compatibility, preflight, formatting, lint,
      type, and diff checks passed. This is static, unit, contract, mocked-
      browser, build, and local HTTP evidence only; no Canny download, repair,
      model execution, generated media, or live workload qualification occurred.
    - [x] `FluxReduxPipeline:edit_image`: backend commit `6be23e7` moves its
      execution profile, capability, Auto resource contract, generic image
      loader, reference-image loader, edit generator, preview route, field
      bindings, and receipt hash `studio-spec-v1-18e2c4ac` into the
      specification registry. Client commit `b709126` removes Redux from the
      Flux-family graph and pipeline-class switches and materializes the exact
      edit recipe from the backend contract. The focused backend matrix passed
      67 tests with 321 subtests; the exact mirrored backend tree passed 1,122
      tests with 4 skips and 1,762 subtests. The focused client graph suite
      passed 43/43, the complete `npm run check` passed, the exact Redux browser
      transition passed 1/1, and the final full mocked Studio browser suite
      passed 87/87. The production bundle was 522627/523264 gzip bytes, 509
      bytes below the stricter 523136-byte safety target. The mirror matched all
      26 generated files byte-for-byte while preserving 317 backend-owned
      Gallery files, and the fresh local HTTP smoke returned 200 for the index
      and all seven referenced assets. Ruff E9/F, package compatibility,
      preflight, formatting, lint, type, and diff checks passed. This is static,
      unit, contract, mocked-browser, build, and local HTTP evidence only; no
      Redux download, model execution, generated media, or live workload
      qualification occurred.
    - [x] `WanTI2VPipeline:text_to_video`: backend commit `276dd1f` moves its
      execution profile, capability, Auto resource contract, generic video
      quantization, execution-recipe, pipeline, generation, and export roles,
      field bindings, and receipt hash `studio-spec-v1-da22e734` into the
      specification registry. Client commit `049addb` removes the TI2V model
      from the legacy video pipeline, artifact, native-flash, and scheduler
      switches and materializes the five-node recipe from the backend contract.
      Follow-up backend `6983ce6` and client `60f4036` restore the prior video
      node coordinates and clear native-flash component selection on CPU; those
      corrections are covered by the subsequent I2V full-gate evidence below.
      The original focused backend matrix passed 67 tests with 325 subtests;
      the exact pre-mirror backend tree passed 1,122 tests with 4 skips and
      1,766 subtests. The focused client specification test and exact mocked-
      browser transition each passed 1/1, the complete `npm run check` passed,
      and the final full mocked Studio browser suite passed 87/87. The
      production bundle was 522709/523264 gzip bytes, 427 bytes below the
      stricter 523136-byte safety target. The mirror matched all 26 generated
      files byte-for-byte while preserving 317 backend-owned Gallery files,
      and a fresh local HTTP smoke returned 200 for the index and all eight
      requested generated asset references. Ruff E9/F, package compatibility,
      preflight, formatting, lint, type, and diff checks passed. This is static,
      unit, contract, mocked-browser, build, and local HTTP evidence only; no
      Wan model download, model execution, generated media, or live workload
      qualification occurred.
    - [x] `WanImageToVideoPipeline:image_to_video`: backend commit `6983ce6`
      moves its execution profile, capability, Auto resource contract, generic
      video runtime/loader/generator/export roles, generic opening-image loader,
      exact image edge, dual-transformer bindings, and receipt hash
      `studio-spec-v1-fed2321d` into the specification registry. Client commit
      `60f4036` removes the I2V class, artifact, VAE-tiling, quantization, and
      native-flash branches from `graphBridge` while preserving the separate
      restored Modular-video route. The focused backend matrix passed 68 tests
      with 329 subtests; the exact pre-mirror backend tree passed 1,123 tests
      with 4 skips and 1,770 subtests. The focused client graph contract passed
      1/1, the I2V and complete eight-spec mocked-browser transitions passed
      2/2, the complete `npm run check` passed, and the final full mocked Studio
      browser suite passed 87/87. The production bundle was 522682/523264 gzip
      bytes, 454 bytes below the stricter 523136-byte safety target. The mirror
      matched all 26 generated files byte-for-byte while preserving 317 backend-
      owned Gallery files. A fresh worker returned 200 for the index and all
      eight requested assets and published both Wan hashes across eight exact
      specifications before its process stopped and port 8088 became free.
      Ruff E9/F, `py_compile`, package compatibility, preflight, formatting,
      lint, type, and diff checks passed. This is static, unit, contract,
      mocked-browser, build, and local HTTP evidence only; no I2V model download,
      model execution, generated media, or live workload qualification occurred.
    - [x] `WanVideoPipeline:text_to_video`: backend commit `92cd1f5`
      moves the exact `wan-text-to-video:direct` profile, mode-specific Auto
      requirements, five-node generic video recipe, declarative form bindings,
      and receipt hash `studio-spec-v1-10c9a3f2` into the specification registry.
      Client commit `0e359ce` adds the exact bounded
      `studioExecutionSpecModes` ownership contract, materializes the migrated
      mode from the backend recipe, and removes the now-unreachable legacy
      `WanPipeline` construction/native-flash switches. The marker/spec mode sets
      must match exactly, so a missing, duplicate, unknown, or mismatched claimed
      mode fails closed; the unclaimed `video_to_video` and `video_color_edit`
      siblings retain their prior `WanVideoToVideoPipeline` graph path. The
      focused backend matrix passed 69 tests with 329 subtests; the exact mirrored
      backend tree passed 1,124 tests with 4 skips, the existing Diffusers
      deprecation warning, and 1,770 subtests. The focused client parser/graph
      matrix passed 70/70, the exact mocked-browser recipe and sibling-mode
      transition passed 1/1, the complete `npm run check` passed, and the final
      full mocked Studio browser suite passed 87/87. The production bundle was
      522738/523264 gzip bytes, 398 bytes below the stricter 523136-byte safety
      target. The mirror matched all 26 generated files byte-for-byte while
      preserving 317 backend-owned Gallery files. A fresh worker returned 200
      for the index and all 23 generated assets, published nine exact specs and
      the Wan marker/hash, and port 8088 was free after the worker stopped. Ruff
      0.12.7 E9/F, `py_compile`, `uv pip check` (78 packages), preflight,
      formatting, lint, type, and diff checks passed. This is static, unit,
      contract, mocked-browser, build, and local HTTP evidence only; no Wan model
      download, model execution, generated media, or new live workload
      qualification occurred.
    - [x] `WanVideoPipeline:video_to_video`: backend commit `93b1e17`
      moves the existing `wan-video-to-video:direct` profile, seven-role generic
      video-input/normalization recipe, exact source-video and dimension/frame
      bindings, and receipt hash `studio-spec-v1-473c930e` into the specification
      registry. Client commit `651eeb3` extends the bounded generic role/source
      vocabulary and materializes the exact V2V topology from that receipt. The
      sibling `video_color_edit` mode remains deliberately unclaimed on its
      legacy graph path. The focused backend matrix passed 131 tests with 348
      subtests; the exact mirrored backend tree passed 1,127 tests with 4 skips,
      the existing Diffusers deprecation warning, and 1,770 subtests. The
      focused client graph contract and exact mocked-browser transition each
      passed 1/1, the complete `npm run check` passed, and the final full mocked
      Studio browser suite passed 87/87 in 219 seconds. The production bundle
      was 522745/523264 gzip bytes, 391 bytes below the stricter 523136-byte
      safety target. The mirror matched all 26 generated files byte-for-byte
      while preserving 317 backend-owned Gallery files. A fresh supervised
      server returned 200 for `/`, the favicon, and all 23 generated assets,
      published seventeen exact specs with Wan ownership limited to
      `text_to_video` and `video_to_video`, and exposed the exact V2V hash while
      omitting a color-edit receipt. Its verified five-process supervisor and
      worker tree stopped and port 8088 was free. Ruff 0.12.7 E9/F,
      `py_compile`, `uv pip check` (78 packages), preflight, formatting, lint,
      type, and diff checks passed. This is static, unit, contract,
      mocked-browser, build, and local HTTP evidence only; no Wan model
      download, model execution, generated media, or live workload
      qualification occurred.
    - [x] `WanVideoPipeline:video_color_edit`: backend commit `09b1d4b`
      seals the final Wan 2.1 sibling with the same reviewed seven-role V2V
      recipe and a distinct receipt hash `studio-spec-v1-0be460bc`. Client
      commit `2525937` materializes that exact receipt and removes the remaining
      `WanVideoPipeline` class, repository, and scheduler branches from
      `graphBridge`; all three Wan 2.1 modes are now specification-owned. The
      focused backend matrix passed 131 tests with 348 subtests; the exact
      mirrored backend tree passed 1,127 tests with 4 skips, the existing
      Diffusers deprecation warning, and 1,770 subtests. The focused client graph
      contract and exact mocked-browser transition each passed 1/1, the complete
      `npm run check` passed, and the complete mocked Studio browser suite passed
      87/87 in 219.2 seconds. The production bundle was 522662/523264 gzip bytes,
      474 bytes below the stricter 523136-byte safety target. The mirror matched
      all 26 generated files byte-for-byte while preserving 317 backend-owned
      Gallery files. A fresh supervised server returned 200 for `/`, the favicon,
      and all 23 generated assets and published eighteen exact specs with all
      three Wan modes; the V2V and color-edit hashes were distinct while sharing
      the reviewed profile. Its verified five-process supervisor and worker tree
      stopped and port 8088 was free. Ruff 0.12.7 E9/F, `py_compile`,
      `uv pip check` (78 packages), preflight, formatting, lint, type, and diff
      checks passed. This is static, unit, contract, mocked-browser, build, and
      local HTTP evidence only; no Wan model download, model execution,
      generated media, or live workload qualification occurred.
    - [x] `LTXVideoPipeline:text_to_video`: backend commit `7736dd3` moves
      the existing four-mode `ltx-video:direct` profile and the text-to-video
      generic video recipe into the versioned specification registry with
      receipt `studio-spec-v1-8f100d39`. Client commit `9f2122f` materializes
      the exact `LTXConditionPipeline` loader, portable native-math attention,
      and LTX-specific generation bindings without adding a model-named graph
      branch. Exact ownership remains limited to `text_to_video`; the sibling
      image-, video-, and reference-to-video modes retain their prior paths.
      The focused backend matrix passed 132 tests with 348 subtests; the exact
      mirrored backend tree passed 1,128 tests with 4 skips, the existing
      Diffusers deprecation warning, and 1,770 subtests. The focused client
      graph contract and exact mocked-browser transition each passed 1/1, the
      complete `npm run check` passed, and the final full mocked Studio browser
      suite passed 87/87 in 220.8 seconds. The production bundle was
      522678/523264 gzip bytes, 458 bytes below the stricter 523136-byte safety
      target. The mirror matched all 26 generated files byte-for-byte while
      preserving 317 backend-owned Gallery files. A fresh supervised server
      returned 200 for `/`, the favicon, and all 23 generated assets, published
      nineteen exact specs with only the LTX text mode claimed, and exposed the
      exact LTX receipt and class. Its verified five-process tree stopped and
      port 8088 was free. Ruff 0.12.7 E9/F, `py_compile`, `uv pip check` (78
      packages), preflight, formatting, lint, type, and diff checks passed.
      This is static, unit, contract, mocked-browser, build, and local HTTP
      evidence only; no LTX model download, model execution, generated media,
      or live workload qualification occurred.
    - [x] `LTXVideoPipeline:video_to_video`: backend commit `e83c760`
      adds the reviewed load/normalize video route and binds source-trajectory
      `strength` to `conditioningScale` while preserving form `strength` as the
      independent `denoise_strength`. Receipt `studio-spec-v1-ad97d224` keeps
      the shared `LTXConditionPipeline`, portable attention, and no-scheduler-
      shift contract. Client commit `8bd95e6` proves the exact topology and both
      values through generic specification materialization. Exact LTX ownership
      now covers text-, image-, and video-to-video; reference-to-video remains
      unclaimed. The focused backend matrix passed 132 tests with 348 subtests;
      the exact mirrored backend tree passed 1,128 tests with 4 skips, the
      existing Diffusers deprecation warning, and 1,770 subtests. The focused
      client graph contract and exact mocked-browser transition each passed
      1/1, the complete `npm run check` passed, and the full mocked Studio
      browser suite passed 87/87 in 221.1 seconds. The unchanged production
      bundle was 522678/523264 gzip bytes, 458 bytes below the stricter
      523136-byte safety target. The mirror matched all 26 generated files
      byte-for-byte while preserving 317 backend-owned Gallery files. A fresh
      supervised server returned 200 for `/`, the favicon, and all 23 generated
      assets, published twenty-one exact specs and all three expected LTX
      hashes, and exposed both distinct strength bindings. Its verified five-
      process tree stopped and port 8088 was free. Ruff 0.12.7 E9/F,
      `py_compile`, `uv pip check` (78 packages), preflight, formatting, lint,
      type, and diff checks passed. This is static, unit, contract, mocked-
      browser, build, and local HTTP evidence only; no LTX model download,
      model execution, generated media, or live workload qualification
      occurred.
    - [x] `LTXVideoPipeline:image_to_video`: backend commit `7b8d5c1`
      adds the exact image sibling over the reviewed `ltx-video:direct` profile
      with the generic source-image role, `image -> reference_images` edge,
      reference list/alpha bindings, and receipt `studio-spec-v1-71f17ad0`.
      Client commit `709ddd3` proves that the generic specification path
      preserves portable native-math attention and LTX generation behavior
      without inheriting Wan dual-transformer, forced-tiling, native-flash, or
      scheduler bindings. Exact LTX ownership is now text- and image-to-video;
      video- and reference-to-video remain deliberately unclaimed. The focused
      backend matrix passed 132 tests with 348 subtests; the exact mirrored
      backend tree passed 1,128 tests with 4 skips, the existing Diffusers
      deprecation warning, and 1,770 subtests. The focused client graph contract
      and exact mocked-browser transition each passed 1/1, the complete
      `npm run check` passed, and the full mocked Studio browser suite passed
      87/87 in 222.4 seconds. The unchanged production bundle was
      522678/523264 gzip bytes, 458 bytes below the stricter 523136-byte safety
      target. The mirror matched all 26 generated files byte-for-byte while
      preserving 317 backend-owned Gallery files. A fresh supervised server
      returned 200 for `/`, the favicon, and all 23 generated assets, published
      twenty exact specs with only the two migrated LTX modes claimed, and
      exposed both expected hashes. Its verified five-process tree stopped and
      port 8088 was free. Ruff 0.12.7 E9/F, `py_compile`, `uv pip check` (78
      packages), preflight, formatting, lint, type, and diff checks passed.
      This is static, unit, contract, mocked-browser, build, and local HTTP
      evidence only; no LTX model download, model execution, generated media,
      or live workload qualification occurred.
    - [x] `LTXVideoPipeline:reference_to_video`: backend commit `0bdc364`
      seals the fourth LTX mode with the reviewed multi-image reference route
      and distinct receipt `studio-spec-v1-0c5abd50`, reusing the exact generic
      image role, edge, and bindings rather than adding a mode-specific node.
      Client commit `cddd140` proves multiple reference paths, alpha handling,
      portable attention, and receipt selection through the shared
      materializer. All four LTX modes are now specification-owned. The focused
      backend matrix passed 132 tests with 348 subtests; the exact mirrored
      backend tree passed 1,128 tests with 4 skips, the existing Diffusers
      deprecation warning, and 1,770 subtests. The focused client graph contract
      and exact mocked-browser transition each passed 1/1, the complete
      `npm run check` passed, and the full mocked Studio browser suite passed
      87/87 in 221.3 seconds. The unchanged production bundle was
      522678/523264 gzip bytes, 458 bytes below the stricter 523136-byte safety
      target. The mirror matched all 26 generated files byte-for-byte while
      preserving 317 backend-owned Gallery files. A fresh supervised server
      returned 200 for `/`, the favicon, and all 23 generated assets, published
      twenty-two exact specs, and exposed all four LTX modes, hashes, and the
      single reviewed pipeline class. Its verified five-process tree stopped
      and port 8088 was free. Ruff 0.12.7 E9/F, `py_compile`, `uv pip check`
      (78 packages), preflight, formatting, lint, type, and diff checks passed.
      This is static, unit, contract, mocked-browser, build, and local HTTP
      evidence only; no LTX model download, model execution, generated media,
      or live workload qualification occurred.
    - [x] `AceStepAudioPipeline:text_to_audio`: backend commit `adcaf48`
      moves the existing `ace-step-audio:direct` profile and reviewed ACE-Step
      repositories into the specification registry, then seals the generic
      quantization, recipe, audio loader, generator, and exporter route with
      receipt `studio-spec-v1-4bc8ed64`. Client commit `5f91ed9` materializes
      the exact `AceStepPipeline` text-to-music recipe, form values, 48 kHz
      generation/export contract, and template base-model override through the
      shared specification path without adding a model-name graph branch.
      Exact ownership at this checkpoint was limited to `text_to_audio`;
      variation joined it in the immediately following migration, while
      continuation and repaint remained on their existing unclaimed paths. The
      focused backend matrix passed 245 tests with 588 subtests; the exact
      mirrored backend tree passed 1,129 tests with 4 skips, the existing
      Diffusers deprecation warning, and 1,770 subtests. The focused client
      graph contract and exact mocked-browser transition each passed 1/1, the
      complete `npm run check` passed, and the complete mocked Studio browser
      suite passed 87/87 in 221 seconds. The production bundle was
      522816/523264 gzip bytes, 320 bytes below the stricter 523136-byte safety
      target. The mirror matched all 26 generated files byte-for-byte while
      preserving 317 installer-owned Gallery files. A fresh supervised server
      returned 200 with byte-exact content for `/`, the favicon, and all 23
      generated assets, published twenty-three exact specs with only ACE-Step
      text-to-audio claimed, and exposed the reviewed receipt and pipeline
      class. Its verified six-process tree stopped and port 8088 was free. Ruff
      0.12.7 E9/F, `py_compile`, `uv pip check` (78 packages), preflight,
      formatting, lint, type, and diff checks passed. This is static, unit,
      contract, mocked-browser, build, and local HTTP evidence only; no ACE-Step
      model download, model execution, generated audio, or live workload
      qualification occurred.
    - [x] `AceStepAudioPipeline:audio_variation`: backend commit `5f6ffdc`
      adds the source-audio loader, exact `audio -> source_audio` edge, `cover`
      task binding, and distinct receipt `studio-spec-v1-eb222623` over the
      same reviewed `ace-step-audio:direct` profile. Client commit `2a776c0`
      extends the bounded generic role/source vocabulary and proves source-file
      binding, exact topology, task selection, and receipt materialization
      without adding an ACE-Step mode branch. Exact ACE-Step ownership at this
      checkpoint covered text generation and variation; continuation joined it
      in the immediately following migration, while repaint remained
      deliberately unclaimed. The focused backend matrix passed 245 tests with
      588 subtests; the exact mirrored backend tree passed 1,129 tests with 4
      skips, the existing Diffusers deprecation warning, and 1,770 subtests.
      The focused client graph contract and exact mocked-browser transition
      each passed 1/1, the complete `npm run check` passed, and the complete
      mocked Studio browser suite passed 87/87 in 221.5 seconds. The production
      bundle was 522832/523264 gzip bytes, 304 bytes below the stricter
      523136-byte safety target. The mirror matched all 26 generated files
      byte-for-byte while preserving 317 installer-owned Gallery files. A
      fresh supervised server returned 200 with byte-exact content for `/`, the
      favicon, and all 23 generated assets, published twenty-four exact specs,
      and exposed exactly the text and variation ACE-Step modes plus the new
      receipt. Its owned process tree stopped and port 8088 was free. Ruff
      0.12.7 E9/F, `py_compile`, `uv pip check` (78 packages), preflight,
      formatting, lint, type, and diff checks passed. This is static, unit,
      contract, mocked-browser, build, and local HTTP evidence only; no ACE-Step
      model download, model execution, generated audio, or live workload
      qualification occurred.
    - [x] `AceStepAudioPipeline:audio_continuation`: backend commit `10b9b1c`
      adds the exact source-audio route, `continuation` task and tail binding,
      generic loudness-match and audio-join roles, reviewed loudness/fade
      constants, and receipt `studio-spec-v1-541adefc` over the same
      `ace-step-audio:direct` profile. Client commit `7e69367` extends only the
      bounded generic role/source vocabulary and proves the exact
      `Load -> Generate -> MatchLoudness -> Join -> Export` topology, values,
      readiness, and receipt without adding an ACE-Step mode branch. Exact
      ACE-Step ownership at this checkpoint covered text generation, variation,
      and continuation; repaint joined them in the immediately following
      migration. The focused
      backend matrix passed 245 tests with 588 subtests; the exact mirrored
      backend tree passed 1,129 tests with 4 skips, the existing Diffusers
      deprecation warning, and 1,770 subtests. The complete client graph suite
      passed 43/43, the exact mocked-browser transition passed 1/1, the complete
      `npm run check` passed, and the complete mocked Studio browser suite
      passed 87/87 in 3.7 minutes. The production bundle was
      522943/523264 gzip bytes, 193 bytes below the stricter 523136-byte safety
      target. The mirror matched all 26 generated files byte-for-byte while
      preserving 317 installer-owned Gallery files. A fresh supervised server
      served byte-exact content for `/`, the favicon, and all 23 generated
      assets, published twenty-five exact specs with exactly the three claimed
      ACE-Step modes, and exposed the continuation receipt. Its verified six-
      process tree stopped and port 8088 was free. Ruff 0.12.7 E9/F,
      `py_compile`, `uv pip check` (78 packages), preflight, formatting, lint,
      type, and diff checks passed. This is static, unit, contract, mocked-
      browser, build, and local HTTP evidence only; no ACE-Step model download,
      model execution, generated audio, or live workload qualification
      occurred.
    - [x] `AceStepAudioPipeline:audio_repaint`: backend commit `60b87a1`
      seals the final ACE-Step sibling with the generic source-audio route,
      exact `repaint` task and repaint-range bindings, and distinct receipt
      `studio-spec-v1-8f5c37c7` over the same reviewed
      `ace-step-audio:direct` profile. Client commit `62dfe17` adds only the
      bounded `repaint` binding source and proves the source-file, task, range,
      topology, readiness, and receipt through the shared specification
      materializer. All four ACE-Step modes are now specification-owned. The
      focused backend matrix passed 245 tests with 588 subtests; the exact
      mirrored backend tree passed 1,129 tests with 4 skips, the existing
      Diffusers deprecation warning, and 1,770 subtests. The complete client
      graph suite passed 43/43, the exact mocked-browser transition passed 1/1,
      the complete `npm run check` passed, and the complete mocked Studio
      browser suite passed 87/87 in 3.7 minutes. The production bundle was
      522950/523264 gzip bytes, 186 bytes below the stricter 523136-byte safety
      target. The mirror matched all 26 generated files byte-for-byte while
      preserving 317 installer-owned Gallery files. A fresh supervised server
      served byte-exact content for `/`, the favicon, and all 23 generated
      assets, published twenty-six exact specs with all four ACE-Step modes,
      and exposed the repaint receipt. Its verified six-process tree stopped
      and port 8088 was free. Ruff 0.12.7 E9/F, `py_compile`, `uv pip check`
      (78 packages), preflight, formatting, lint, type, and diff checks passed.
      This is static, unit, contract, mocked-browser, build, and local HTTP
      evidence only; no ACE-Step model download, model execution, generated
      audio, or live workload qualification occurred.
    - [x] `QwenImageEditModularPipeline:inpaint`: backend commit `de2160f`
      seals the unambiguous `qwen-edit:direct-inpaint` profile, reviewed
      `QwenImageEditInpaintPipeline`, source-image and mask loaders, generic
      inpaint/preview route, and exact form bindings with receipt
      `studio-spec-v1-ac52abb3`. Client commit `f28ff89` materializes that
      contract through the shared specification path and removes three inert
      model-named direct-Qwen diagnostic predicates. Exact ownership remains
      limited to inpaint; the distinct outpaint and Modular edit recipes remain
      deliberately unclaimed. The focused backend matrix passed 246 tests with
      588 subtests; the exact mirrored backend tree passed 1,130 tests with 4
      skips, the existing Diffusers deprecation warning, and 1,770 subtests.
      The complete client graph suite passed 43/43, the exact mocked-browser
      transition passed 1/1, the complete `npm run check` passed, and the
      complete mocked Studio browser suite passed 87/87 in 3.7 minutes. The
      production bundle remained 522950/523264 gzip bytes, 186 bytes below the
      stricter 523136-byte safety target. The mirror matched all 26 generated
      files byte-for-byte while preserving 317 installer-owned Gallery files.
      A fresh supervised server served byte-exact content for `/`, the favicon,
      and all 23 generated assets, published twenty-seven exact specs, exposed
      only the Qwen Image Edit inpaint marker/receipt, and kept outpaint
      unclaimed. Its verified six-process tree stopped and port 8088 was free.
      Ruff 0.12.7 E9/F, `py_compile`, `uv pip check` (78 packages), preflight,
      formatting, lint, type, and diff checks passed. This is static, unit,
      contract, mocked-browser, build, and local HTTP evidence only; no Qwen
      model download, model execution, generated image, or live workload
      qualification occurred.
    - [x] `WanVACEPipeline:text_to_video`: backend commit `69a8561`, corrected
      by `2616014` and mirrored by `69247d0`,
      seals the reviewed `wan-vace:direct` profile, immutable
      `Wan-AI/Wan2.1-VACE-1.3B-diffusers` artifact, shared generic video
      recipe, exact text-mode bindings, and receipt
      `studio-spec-v1-4a34e319`. Client commit `8f05541`, corrected by
      `4c9d40c`, proves that the
      existing specification materializer builds the exact four-edge route,
      pipeline identity, artifact, mode, readiness, and receipt without a new
      model-named production branch. The correction binds and persists the
      catalog's reviewed `ec4d2cb062b548996b179d493fdd05340de702a1`
      revision instead of relying only on the loader's execution-time catalog
      resolution. At this checkpoint ownership remained limited to
      text-to-video; VACE inpaint, outpaint, and control modes were
      deliberately unclaimed until their distinct conditioned-input graphs
      were migrated. The focused backend matrix passed 188 tests with 360
      subtests; the exact mirrored backend tree passed 1,131 tests with 4
      skips, the existing Diffusers deprecation warning, and 1,770 subtests.
      The complete client graph suite passed 43/43, the exact mocked-browser
      transition passed 1/1, the complete `npm run check` passed, and the final
      exact-tree mocked Studio browser suite passed 87/87 in 3.7 minutes. The
      production bundle was 522970/523264 gzip bytes, 166 bytes below the
      stricter 523136-byte safety target. The mirror matched all 26 generated
      files byte-for-byte while preserving 317 installer-owned Gallery files.
      A fresh supervised server served byte-exact content for `/`, the
      favicon, and all 23 generated assets, published twenty-eight exact
      specs, exposed only the Wan VACE text-to-video marker/receipt, and kept
      its three conditioned siblings unclaimed. Its verified six-process tree
      stopped and port 8088 was free. Ruff 0.12.7 E9/F, `py_compile`, `uv pip
      check` (78 packages), preflight, formatting, lint, type, and diff checks
      passed. This is static, unit, contract, mocked-browser, build, and local
      HTTP evidence only; no Wan VACE model download, model execution,
      generated video, or live workload qualification occurred.
    - [x] `WanVACEPipeline:video_inpaint`: backend and bundled-client commit
      `0e9c3f2` extends the reviewed `wan-vace:direct` specification with the
      exact source-video normalization and aligned-mask route, immutable VACE
      revision binding, source and mask file bindings, and reviewed threshold
      127 / 96-pixel inpaint mask-growth policy. Client commit `3f79ca2`
      accepts only those new generic roles and binding sources, materializes
      the exact nine-role/nine-edge recipe, and verifies receipt
      `studio-spec-v1-d0b56303`, both media inputs, mask policy, mode, and Run
      readiness without a model-named production branch. At this checkpoint
      ownership was limited to `video_inpaint`; VACE outpaint and
      control-to-video remained unclaimed until their distinct conditioned-
      input contracts were migrated. The
      focused backend matrix passed 189 tests with 360 subtests; the complete
      backend gate passed 1,132 tests with 4 skips, the existing Diffusers
      deprecation warning, and 1,770 subtests. The client graph suite passed
      43/43, the exact mocked-browser transition passed 1/1, the complete
      `npm run check` passed, and the frozen complete mocked Studio suite
      passed 87/87 in 3.6 minutes. The production bundle was
      523045/523264 gzip bytes, 91 bytes below the stricter 523136-byte safety
      target. The mirror matched all 26 generated files byte-for-byte while
      preserving 317 installer-owned Gallery files. A fresh backend served
      byte-exact content for all 25 public generated files, published 29 exact
      specs, and exposed exactly the VACE text-to-video and video-inpaint
      markers with the inpaint revision source and mask route intact. Its
      verified three-process tree stopped and port 8088 was free. Ruff 0.12.7
      E9/F, `py_compile`, `uv pip check` (78 packages), preflight, formatting,
      lint, type, and diff checks passed. This is static, unit, contract,
      mocked-browser, build, and local HTTP evidence only; no Wan VACE model
      download, model execution, generated video, or live workload
      qualification occurred.
    - [x] `WanVACEPipeline:video_outpaint`: backend and bundled-client commit
      `b004af1` adds the distinct outpaint receipt over the reviewed VACE
      source-normalization and aligned-boundary-mask route. It retains the
      immutable VACE artifact revision and threshold 127 while binding the
      legacy outpaint policy to zero mask growth. Client commit `fce224e`
      materializes that same nine-role/nine-edge graph, seals receipt
      `studio-spec-v1-1fd16911`, and verifies both media inputs, the distinct
      zero-growth policy, mode, and Run readiness without a model-named
      production branch. Ownership is limited to `video_outpaint`; VACE
      control-to-video remains unclaimed until its distinct control-input
      contract is migrated. The focused backend matrix passed 190 tests with
      360 subtests; the complete backend gate passed 1,133 tests with 4 skips,
      the existing Diffusers deprecation warning, and 1,770 subtests. The
      client graph suite passed 43/43, the exact mocked-browser transition
      passed 1/1, the complete `npm run check` passed, and the frozen complete
      mocked Studio suite passed 87/87 in 3.6 minutes. The production bundle
      was 523055/523264 gzip bytes, 81 bytes below the stricter 523136-byte
      safety target. The mirror matched all 26 generated files byte-for-byte
      while preserving 317 installer-owned Gallery files. A fresh backend
      served byte-exact content for all 25 public generated files, published
      30 exact specs, and exposed exactly the VACE text-to-video, video-
      inpaint, and video-outpaint markers with the distinct outpaint growth
      source intact. Its verified five-process tree stopped and port 8088 was
      free. Ruff 0.12.7 E9/F, `py_compile`, `uv pip check` (78 packages),
      preflight, formatting, lint, type, and diff checks passed. This is
      static, unit, contract, mocked-browser, build, and local HTTP evidence
      only; no Wan VACE model download, model execution, generated video, or
      live workload qualification occurred.
    - [x] `WanVACEPipeline:control_to_video`: backend and bundled-client commit
      `ed07f34` completes the exact specification coverage for all four
      advertised VACE modes. It binds the reviewed direct loader and immutable
      VACE revision to a separate control-video loader, width/height/frame-count
      normalization, generator `video` input, and the existing generic export
      route; it does not admit the source-video or mask branches. Client commit
      `72ed446` accepts only the added generic control-video role and form source,
      materializes the exact seven-role/six-edge graph, seals receipt
      `studio-spec-v1-d05d263d`, and verifies the control file, normalized frame
      count, mode, absence of source/mask roles, and Run readiness without a
      model-named production branch. The focused backend matrix passed 191 tests
      with 360 subtests; the complete backend gate passed 1,134 tests with 4
      skips, the existing Diffusers deprecation warning, and 1,770 subtests in
      45.08 seconds. The client graph suite passed 43/43, the exact mocked-browser
      transition passed 1/1, the complete `npm run check` passed, and the frozen
      complete mocked Studio suite passed 87/87 in 220.1 seconds. The production
      bundle was 523065/523264 gzip bytes, 71 bytes below the stricter
      523136-byte safety target. The mirror matched all 26 generated files
      byte-for-byte while preserving 317 installer-owned Gallery files. A fresh
      supervised server served byte-exact content for all 25 public generated
      files, published 31 exact specs, and exposed exactly all four advertised
      VACE markers with the control receipt, topology, and bindings intact. Its
      verified five-process tree stopped and port 8088 was free. Ruff 0.12.7
      E9/F, `py_compile`, `uv pip check` (78 packages), preflight, formatting,
      lint, type, and diff checks passed. This is static, unit, contract,
      mocked-browser, build, and local HTTP evidence only; no Wan VACE model
      download, model execution, generated video, or live workload qualification
      occurred.
    - [x] `QwenImageEditModularPipeline:outpaint`: backend and bundled-client
      commit `823357d` extends the reviewed `qwen-edit:direct-inpaint` profile
      with the distinct generated-canvas route, removes the separate mask-file
      loader from this mode, and seals every boundary-placement binding with
      receipt `studio-spec-v1-4ffd900b`. Client commit `de2eba1` accepts only
      the existing generic outpaint-canvas role and seven form sources,
      materializes the exact seven-role/seven-edge source-to-canvas-to-inpaint
      graph, and verifies the source file, canvas dimensions and offsets, mask
      route, absence of `loadMask`, receipt, and Run readiness without a new
      model-named production branch. Qwen Image Edit inpaint and outpaint are
      now specification-owned; its distinct Modular edit recipe remains
      deliberately unclaimed. The focused backend matrix passed 192 tests with
      360 subtests; the complete backend gate passed 1,135 tests with 4 skips,
      the existing Diffusers deprecation warning, and 1,770 subtests in 41.89
      seconds. The client graph suite passed 43/43, the exact mocked-browser
      transition passed 1/1, the complete `npm run check` passed, and the
      frozen complete mocked Studio suite passed 87/87 in 219.6 seconds. The
      production bundle was 523109/523264 gzip bytes, 27 bytes below the
      stricter 523136-byte safety target. The mirror matched all 26 generated
      files byte-for-byte while preserving 317 installer-owned Gallery files.
      A fresh backend served byte-exact content for all 25 public generated
      files, published 32 exact specs, and exposed both Qwen Image Edit modes
      with the outpaint receipt, topology, and bindings intact. Its verified
      five-process tree stopped and port 8088 was free. Ruff 0.12.7 E9/F,
      `py_compile`, `uv pip check` (78 packages), preflight, formatting, lint,
      type, and diff checks passed. This is static, unit, contract,
      mocked-browser, build, and local HTTP evidence only; no Qwen model
      download, model execution, generated image, or live workload
      qualification occurred.
    - [x] `ZImageModularPipeline:text_to_image`: backend commit `a4efd6c`
      moves the existing direct `z-image:auto` loader profile and generic
      five-node image recipe into exact specification ownership with receipt
      `studio-spec-v1-0d3c1205`. Client commit `77ceab9` proves the existing
      generic materializer consumes that contract without a production source
      change. The complete evidence and remaining six-pair boundary are recorded
      in the P0.4 receipt item below.
    - [x] `QwenImageModularPipeline:text_to_image`: backend commit `6e40bab`
      moves the existing direct `qwen-image:t2i-direct` profile and generic
      five-node image recipe into exact specification ownership with receipt
      `studio-spec-v1-f53ab380`. Client commit `531d4b9` proves the current
      official and reviewed prequantized artifact candidates bind and consume
      the same generic graph contract without a production source change. The
      complete evidence and remaining five-pair boundary are recorded in the
      P0.4 receipt item below.
    - [x] `FluxKontextPipeline:edit_image`: backend commit `5f4d437`
      moves the existing `flux-kontext:direct` profile, exact edit-only Auto
      requirements, six-node generic edit recipe, declarative form bindings,
      and receipt hash `studio-spec-v1-393009a9` into the specification
      registry. Client commit `8ae0dd9` proves the generic specification path
      without adding a new model-named production branch. Exact mode ownership
      was limited to `edit_image`; `multi_image_reference_edit` remained on its
      existing generic legacy graph until the immediately following paired
      migration. The focused backend
      matrix passed 98 tests with 451 subtests; the exact mirrored backend tree
      passed 1,125 tests with 4 skips, the existing Diffusers deprecation
      warning, and 1,770 subtests. The focused client graph contract and exact
      mocked-browser recipe/sibling transition each passed 1/1, the complete
      `npm run check` passed, and the final full mocked Studio browser suite
      passed 87/87. The unchanged production bundle was 522738/523264 gzip
      bytes, 398 bytes below the stricter 523136-byte safety target. The mirror
      matched all 26 generated files byte-for-byte while preserving 317
      backend-owned Gallery files. A fresh worker returned 200 for `/`, the
      favicon, and all 23 generated assets, published ten exact specs plus the
      Kontext `edit_image` marker/hash, and port 8088 was free after its verified
      worker stopped. Ruff 0.12.7 E9/F, `py_compile`, `uv pip check` (78
      packages), preflight, formatting, lint, type, and diff checks passed. This
      is static, unit, contract, mocked-browser, build, and local HTTP evidence
      only; no Kontext download, model execution, generated media, or live
      workload qualification occurred.
    - [x] `FluxKontextPipeline:multi_image_reference_edit`: backend commit
      `119c720` adds the second exact receipt
      `studio-spec-v1-aa060039`, reuses the same reviewed
      `flux-kontext:direct` loader profile without making loader-only optional-
      runtime resolution ambiguous, and keeps Auto requirements explicitly
      edit-only. Client commit `d956a42` removes Kontext from the remaining
      Flux-family and pipeline-class switches; both modes now materialize the
      same generic six-node edit topology entirely from their distinct backend
      specifications. The focused backend matrix passed 98 tests with 451
      subtests; the exact mirrored backend tree passed 1,125 tests with 4 skips,
      the existing Diffusers deprecation warning, and 1,770 subtests. The
      focused two-mode graph contract and exact mocked-browser transition each
      passed 1/1, the complete `npm run check` passed, and the final full mocked
      Studio browser suite passed 87/87. The production bundle was
      522724/523264 gzip bytes, 412 bytes below the stricter 523136-byte safety
      target. The mirror matched all 26 generated files byte-for-byte while
      preserving 317 backend-owned Gallery files. A fresh worker returned 200
      for `/`, the favicon, and all 23 generated assets, published eleven exact
      specs plus both Kontext mode markers/hashes, and port 8088 was free after
      its verified worker stopped. Ruff 0.12.7 E9/F, `py_compile`, `uv pip
      check` (78 packages), preflight, formatting, lint, type, and diff checks
      passed. This is static, unit, contract, mocked-browser, build, and local
      HTTP evidence only; no Kontext download, model execution, generated
      media, or live workload qualification occurred.
    - [x] `FluxFillPipeline:inpaint`: backend commit `544c54f` moves the
      shared `flux-fill:direct` execution profile and existing inpaint/outpaint
      Auto resource policy into the versioned specification registry, while
      claiming only the inpaint mode with exact source-image, mask, pipeline,
      and preview edges plus declarative form bindings. Client commit `38f8d81`
      extends the bounded generic parser/materializer for those roles and proves
      receipt `studio-spec-v1-ba8c8dd1`; the sibling `outpaint` mode remains
      explicitly unclaimed on its prior generic graph path. The focused backend
      matrix passed 71 tests with 329 subtests; the exact mirrored backend tree
      passed 1,126 tests with 4 skips, the existing Diffusers deprecation
      warning, and 1,770 subtests. The focused client graph contract and exact
      mocked-browser inpaint/outpaint transition each passed 1/1, the complete
      `npm run check` passed, and the final full mocked Studio browser suite
      passed 87/87. The production bundle was 522756/523264 gzip bytes, 380
      bytes below the stricter 523136-byte safety target. The mirror matched all
      26 generated files byte-for-byte while preserving 317 backend-owned
      Gallery files. A fresh worker returned 200 for `/`, the favicon, and all
      23 generated assets, published twelve exact specs plus the Fill inpaint
      marker/hash, and port 8088 was free after its verified worker stopped.
      Ruff 0.12.7 E9/F, `py_compile`, `uv pip check` (78 packages), preflight,
      formatting, lint, type, and diff checks passed. This is static, unit,
      contract, mocked-browser, build, and local HTTP evidence only; no Fill
      model download, model execution, generated media, or live workload
      qualification occurred.
    - [x] `FluxFillPipeline:outpaint`: backend commit `634c485` adds the
      distinct outpaint receipt `studio-spec-v1-5c0d7413` over the same exact
      reviewed `flux-fill:direct` profile, source-image/mask topology, and
      declarative bindings. Client commit `4c0bd05` proves graph equivalence
      across both Fill modes and removes Fill from the legacy Flux-family and
      pipeline-class switches; neither production graph construction nor
      pipeline selection now branches on `FluxFillPipeline`. The focused
      backend matrix passed 71 tests with 329 subtests; the exact mirrored
      backend tree passed 1,126 tests with 4 skips, the existing Diffusers
      deprecation warning, and 1,770 subtests. The focused client graph contract
      and exact mocked-browser two-mode transition each passed 1/1, the complete
      `npm run check` passed, and the final full mocked Studio browser suite
      passed 87/87. The production bundle was 522741/523264 gzip bytes, 395
      bytes below the stricter 523136-byte safety target. The mirror matched all
      26 generated files byte-for-byte while preserving 317 backend-owned
      Gallery files. A fresh supervised server returned 200 for `/`, the
      favicon, and all 23 generated assets, published thirteen exact specs plus
      both Fill markers/hashes, and port 8088 was free after its verified
      supervisor and worker stopped. Ruff 0.12.7 E9/F, `py_compile`, `uv pip
      check` (78 packages), preflight, formatting, lint, type, and diff checks
      passed. This is static, unit, contract, mocked-browser, build, and local
      HTTP evidence only; no Fill model download, model execution, generated
      media, or live workload qualification occurred.
    - [x] `Flux2KleinPipeline:text_to_image`: backend commit `441cd00`
      moves the shared three-mode `flux2-klein:direct` profile, capability, and
      Auto policy into the versioned specification registry while claiming
      only the text-to-image graph through receipt
      `studio-spec-v1-e11dfdc6`. Client commit `931621d` proves the exact
      generic Diffusers image recipe and stable role topology, while explicitly
      keeping `edit_image` and `multi_image_reference_edit` on their prior
      unclaimed legacy path. The focused backend matrix passed 129 tests with
      329 subtests; the exact mirrored backend tree passed 1,127 tests with 4
      skips, the existing Diffusers deprecation warning, and 1,770 subtests.
      The focused client graph contract and exact mocked-browser transition
      each passed 1/1, the complete `npm run check` passed, and the final full
      mocked Studio browser suite passed 87/87 in 218.2 seconds. The production
      bundle remained 522741/523264 gzip bytes, 395 bytes below the stricter
      523136-byte safety target. The mirror matched all 26 generated files
      byte-for-byte while preserving 317 backend-owned Gallery files. A fresh
      supervised server returned 200 for `/`, the favicon, and all 23 generated
      assets, and published fourteen exact specs plus the Klein marker/hash.
      Its verified supervisor and worker tree stopped and port 8088 was free.
      Ruff 0.12.7 E9/F, `py_compile`, `uv pip check` (78 packages), preflight,
      formatting, lint, type, and diff checks passed. This is static, unit,
      contract, mocked-browser, build, and local HTTP evidence only; no Klein
      model download, model execution, generated media, or live workload
      qualification occurred.
    - [x] `Flux2KleinPipeline:edit_image`: backend commit `ab3bd34`
      adds the distinct edit receipt `studio-spec-v1-ab4da919` over the same
      reviewed `flux2-klein:direct` profile with an exact source-image,
      Diffusers Edit, and preview route. Client commit `84e1d8f` proves the
      exact edit graph and transition while keeping only
      `multi_image_reference_edit` unclaimed. The focused backend matrix passed
      131 tests with 348 subtests; the exact mirrored backend tree passed 1,127
      tests with 4 skips, the existing Diffusers deprecation warning, and 1,770
      subtests. The focused client graph contract and exact mocked-browser
      transition each passed 1/1, the complete `npm run check` passed, and the
      final full mocked Studio browser suite passed 87/87 in 220.1 seconds. The
      production bundle remained 522741/523264 gzip bytes, 395 bytes below the
      stricter 523136-byte safety target. The mirror matched all 26 generated
      files byte-for-byte while preserving 317 backend-owned Gallery files. A
      fresh supervised server returned 200 for `/`, the favicon, and all 23
      generated assets, and published fifteen exact specs plus both Klein
      markers/hashes. Its verified supervisor and worker tree stopped and port
      8088 was free. Ruff 0.12.7 E9/F, `py_compile`, `uv pip check` (78
      packages), preflight, formatting, lint, type, and diff checks passed. This
      is static, unit, contract, mocked-browser, build, and local HTTP evidence
      only; no Klein model download, model execution, generated media, or live
      workload qualification occurred.
    - [x] `Flux2KleinPipeline:multi_image_reference_edit`: backend commit
      `4527764` adds the final distinct Klein receipt
      `studio-spec-v1-756c2d69` over the reviewed shared
      `flux2-klein:direct` profile and the exact source-image, Diffusers Edit,
      and preview route. Client commit `7180694` proves that exact receipt and
      graph transition, and removes `Flux2KleinPipeline` from the legacy Flux
      family and pipeline-class switches; all three Klein modes are now
      specification-owned. The focused backend matrix passed 131 tests with
      348 subtests; the exact mirrored backend tree passed 1,127 tests with 4
      skips, the existing Diffusers deprecation warning, and 1,770 subtests.
      The focused client graph contract and exact mocked-browser transition
      each passed 1/1, the complete `npm run check` passed, and the final full
      mocked Studio browser suite passed 87/87 in 219.1 seconds. The production
      bundle was 522725/523264 gzip bytes, 411 bytes below the stricter
      523136-byte safety target. The mirror matched all 26 generated files
      byte-for-byte while preserving 317 backend-owned Gallery files. A fresh
      supervised server returned 200 for `/`, the favicon, and all 23 generated
      assets, and published sixteen exact specs with all three Klein modes and
      the exact multi-reference hash. Its verified five-process supervisor and
      worker tree stopped and port 8088 was free. Ruff 0.12.7 E9/F,
      `py_compile`, `uv pip check` (78 packages), preflight, formatting, lint,
      type, and diff checks passed. This is static, unit, contract,
      mocked-browser, build, and local HTTP evidence only; no Klein model
      download, model execution, generated media, or live workload
      qualification occurred.
    - [x] Exact-pair specification migration: all 39 currently declared
      execution-profile pairs now have a backend-owned Studio specification and
      a tested generic client receipt. This closes the legacy exact-pair gap but
      does not qualify live model execution or the remaining declarative
      overlays.
    - [x] Loader-component requirement overlay. Backend commit `732e15c` adds a
      bounded `loader_component_outputs` contract to reviewed Modular pipeline
      metadata. Wan I2V now declares `image_encoder`; `ModelsLoader` consumes
      the generic list for both strict required-component loading and published
      component receipts, with no Wan class-name branch. Custom sidecars may
      carry only a bounded declarative list and remain `contract_only`; the
      loader accepts only node-supported outputs from the built-in registry.
      The focused Modular/loading matrix passed 253 tests and 440 subtests. The
      complete backend gate passed 1,155 tests with 4 skips and 1,795 subtests,
      with only the existing Diffusers `torch_dtype` deprecation warning. Ruff
      0.12.7 E9/F, `compileall`, `uv pip check` (78 packages), preflight, and
      diff checks passed. This is static, unit, contract, and local preflight
      evidence only; no model download, component load, inference, or generated
      media occurred.
    - [x] Layer-block allowlist overlay. Backend commit `02afc25` moves the six
      existing SDXL, Qwen Image/Edit/Edit Plus, Flux, and Flux Kontext
      model-to-transformer-stack selections onto bounded reviewed pipeline
      metadata. The generic `Layers` node publishes the generated map without
      pipeline-class keys and now requires the connected model signal at both
      dynamic-field creation and execution. Missing/unknown identities,
      duplicate or unlisted paths, and extra injected block configurations fail
      before an upstream guidance object can consume them. The focused schema,
      registry, and route matrix passed 184 tests and 360 subtests. The complete
      backend gate passed 1,156 tests with 4 skips and 1,802 subtests, with only
      the existing Diffusers `torch_dtype` deprecation warning. Ruff 0.12.7
      E9/F, `compileall`, `uv pip check` (78 packages), preflight, and diff
      checks passed. This is static, unit, contract, and local preflight evidence
      only; no model download, inference, guidance execution, or generated media
      occurred.
    - [x] Denoise image-latent dimension overlay. Backend commit `7228c1f`
      replaces the four-class compatibility branch with bounded reviewed
      `denoise_image_latent_dimensions` metadata. Qwen Image Edit/Edit Plus,
      Flux Kontext, and Flux2 Klein retain legacy hidden `height` and `width`
      values when image latents are present; every other registered pipeline
      drops them. Unknown, malformed, duplicate, and unlisted metadata fails
      closed, while custom sidecars remain `contract_only` and cannot authorize
      this built-in execution exception. The focused schema/security matrix
      passed 99 tests and 240 subtests; the wider no-weight Modular route matrix
      passed 221 tests and 491 subtests. The complete backend gate passed 1,158
      tests with 4 skips and 1,821 subtests, with only the existing Diffusers
      `torch_dtype` deprecation warning. Ruff 0.12.7 E9/F, `compileall`, `uv pip
      check` (78 packages), preflight, and diff checks passed. This is static,
      unit, contract, and local preflight evidence only; no model download,
      inference, or generated media occurred.
    - [x] Guider compatibility overlay. Backend commit `51206e6` declares the
      exact bounded Diffusers guider choices on reviewed Modular pipeline
      metadata. SDXL and Qwen Image/Edit/Edit Plus expose the complete reviewed
      set because they also declare layer-stack contracts; Qwen Layered,
      Z-Image, and Wan expose only non-layer guiders; Flux variants expose no
      guider component. The generic Guider selector consumes the generated map
      and revalidates the connected pipeline identity at field refresh and
      execution, so missing, unknown, malformed, or incompatible selections
      fail before constructing an upstream guider. Client commit `d1b2f88`
      preserves scalar values for dynamic single-select options while retaining
      array values for multi-select Layers. The focused backend schema/security
      matrix passed 101 tests and 258 subtests; the wider no-weight Modular route
      matrix passed 223 tests and 509 subtests. The complete backend gate passed
      1,160 tests with 4 skips and 1,839 subtests, with only the existing
      Diffusers `torch_dtype` deprecation warning. The complete client
      `npm run check` passed, the exact model-signal/Guider/Layers mocked-browser
      contract passed 1/1, and the production bundle was 523123/523264 gzip
      bytes, 13 bytes below the stricter 523136-byte safety target. Ruff 0.12.7
      E9/F, `compileall`, `uv pip check` (78 packages), preflight, formatting,
      lint, type, and diff checks passed. This is static, unit, contract,
      mocked-browser, build, and local preflight evidence only; no model
      download, guider execution, inference, or generated media occurred.
    - [x] Scheduler compatibility overlay. Backend commit `6779a19` publishes
      bounded scheduler choices from the pinned official Diffusers component
      contracts. SDXL and Wan expose the 14 replacements shared by their Euler
      or UniPC scheduler compatibility sets; Qwen, Flux, and Z-Image
      flow-matching pipelines expose no unsupported legacy replacement. The
      generic Scheduler node requires the connected reviewed pipeline identity
      during field refresh and execution, verifies the live component class,
      and requires the replacement constructor to return the exact selected
      official scheduler class. Unknown, malformed, duplicate, oversized, LCM,
      TCD, and incompatible selections fail closed. Client commit `140cab2`
      extends the generic signal-relay browser contract across Guider, Layers,
      and Scheduler without a model-named client branch. The focused backend
      schema/security matrix passed 103 tests and 278 subtests; the wider
      no-weight Modular route matrix passed 225 tests and 529 subtests. The
      complete backend gate passed 1,162 tests with 4 skips and 1,859 subtests,
      with only the existing Diffusers `torch_dtype` deprecation warning. The
      complete client `npm run check` passed, the exact mocked-browser contract
      passed 1/1 twice, and the production bundle was 523123/523264 gzip bytes,
      13 bytes below the stricter 523136-byte safety target. Ruff 0.12.7 E9/F,
      `compileall`, `uv pip check` (78 packages), preflight, formatting, lint,
      type, and diff checks passed. This is static, unit, contract,
      mocked-browser, build, and local preflight evidence only; no model
      download, scheduler execution, inference, or generated media occurred.
    - [x] Execution-spec readiness overlay. Client commit `7a02806` centralizes
      exact specification lookup and derives managed loader capability checks
      from the live binding or the backend-advertised loader module/action. It
      replaces the Qwen-specific missing-node branch with a generic check over
      every role in the exact specification. A deliberately opaque future
      loader contract proves offload readiness without a model-name route, and
      a schema-v2 Qwen outpaint browser fixture proves a missing exact role is
      reported from the authoritative contract. The focused client contract
      passed 67/67, the exact specification browser matrix passed 1/1, the
      complete client gate passed, and the final mocked Studio suite passed
      88/88. The production bundle passed at 523080/523264 gzip bytes, including
      56 bytes of headroom against the stricter 523136-byte safety target. This
      is static, unit, contract, mocked-browser, and build evidence only; no
      model download, inference, or generated media occurred.
    - [x] Diffusers audio field-contract overlay. Backend commit `2a98856`
      makes every reviewed audio pipeline/mode contract publish its canonical
      generic `Generate` field overlay, including visibility, required inputs,
      task choices, and duration bounds. The field action reconstructs the
      exact contract and rejects a stored or client-edited overlay before any
      mutation. Client commit `fba496c` removes the duplicate pipeline-class
      visibility switches from managed graph synchronization and soundtrack
      construction; generic signal/action handling now applies the backend-
      authored fields when the selected pipeline or mode changes. The focused
      audio suite passed 59 tests with 257 subtests, the adjacent contract matrix
      passed 115 tests with 377 subtests, and the complete backend gate passed
      1,162 tests with 4 skips and 1,888 subtests with only the existing
      Diffusers deprecation warning. The focused client graph/specification
      matrix passed 110/110, `npm run check` passed, the field-update,
      exact-audio-recipe, and soundtrack-proof browser paths passed 3/3, and the
      complete mocked Studio suite passed 89/89 in 4.5 minutes. The production
      bundle was 522793/523264 gzip bytes, 343 bytes below the stricter
      523136-byte safety target. Ruff E9/F, `py_compile`, `uv pip check` (78
      packages), preflight, formatting, lint, type, and diff checks passed.
      This is static, unit, contract, mocked-browser, build, and local preflight
      evidence only; no model download, audio execution, generated media, or
      live workload qualification occurred.
    - [x] Auto execution-path authority overlay. Client commit `16b7f12`
      removes the pre-plan Qwen and broad family execution-path guesses from
      the local resource fallback. Auto now stays path-neutral until an exact
      selected schema-v2 backend candidate supplies the reviewed loader path;
      Expert likewise describes the editable full graph without claiming a
      model-specific execution path. All current model/mode fallbacks are
      covered by a zero-invented-path contract, while the mocked exact Auto run
      proves `direct-diffusers-image` still reaches the submitted receipt from
      the bound backend candidate. The focused resource/request matrix passed
      79/79, `npm run check` passed, the exact Auto submission browser path
      passed 1/1, and the final complete mocked Studio suite passed 89/89 in
      4.5 minutes. The production bundle was 522524/523264 gzip bytes, 612
      bytes below the stricter 523136-byte safety target. Formatting, lint,
      type, build, bundle, and diff checks passed. This is static, unit,
      contract, mocked-browser, and build evidence only; no model download,
      inference, or generated media occurred.
    - [x] Auto retry-mode authority overlay. Backend commit `a1173db` ignores a
      duplicate client `resourceRetryModes` list in Auto and derives fallback
      offload modes from the selected exact execution profile in canonical
      memory-pressure order; Expert retains its bounded explicit hint. Client
      commit `52e98d2` stops submitting `supportedOffloadModes` and
      `resourceRetryModes` in Auto while preserving the selected candidate and
      candidate-bound retry receipts. The focused backend resource/profile
      matrix passed 182 tests with 349 subtests, and the complete backend gate
      passed 1,163 tests with 4 skips and 1,888 subtests with only the existing
      Diffusers deprecation warning. The focused client request/resource matrix
      passed 79/79, `npm run check` passed, the exact Auto submission browser
      path passed 1/1, and the complete mocked Studio suite passed 89/89 in 273
      seconds. The production bundle was 522530/523264 gzip bytes, 606 bytes
      below the stricter 523136-byte safety target. Ruff 0.12.7 E9/F,
      `py_compile`, `uv pip check` (78 packages), preflight, formatting, lint,
      type, and diff checks passed. This is static, unit, contract,
      mocked-browser, build, and local preflight evidence only; no model
      download, inference, generated media, or live workload qualification
      occurred.
    - [x] Runtime diagnostic-classifier cleanup. Backend commit `85e80f3`
      removes client `modelFamily` and `lowVramMode` from the admitted runtime
      hint contract and its CUDA diagnostic projection; exact model type,
      execution profile, selected recipe, and graph receipt remain the reviewed
      execution identities. Client commit `e464cb7` deletes the Qwen-named
      low-memory classifier and stops submitting both duplicate labels. The
      focused backend runtime/resource matrix passed 177 tests with 307
      subtests, and the complete backend gate passed 1,164 tests with 4 skips
      and 1,888 subtests with only the existing Diffusers deprecation warning.
      The focused client template/provenance/run matrix passed 123/123,
      `npm run check` passed, the exact Auto submission browser path passed 1/1,
      and the complete mocked Studio suite passed 89/89 in 280 seconds. The
      production bundle was 522389/523264 gzip bytes, 747 bytes below the
      stricter 523136-byte safety target. Ruff 0.12.7 E9/F, `py_compile`, `uv
      pip check` (78 packages), preflight, formatting, lint, type, and diff
      checks passed. This is static, unit, contract, mocked-browser, build, and
      local preflight evidence only; no model download, inference, generated
      media, or live workload qualification occurred.
    - [x] Diffusers video field-contract overlay. Backend commit `b32241b`
      declares a complete reviewed field contract for every registered generic
      video adapter/mode pair. The existing exact pipeline signal now drives
      mode choices, input visibility and requiredness, adapter-specific
      controls, and the shared strength control's declarative Studio binding;
      the action reconstructs the canonical signal and rejects a stale or
      edited contract before any field mutation. Client commit `947f7d9`
      expands the bounded identity-binding grammar only to the reviewed
      `strength` and `conditioningScale` form fields and removes the remaining
      LTX model-name branch from managed control synchronization. The focused
      video suite passed 79 tests with 161 subtests, the adjacent backend matrix
      passed 115 tests with 208 subtests, and the complete backend gate passed
      1,165 tests with 4 skips and 1,902 subtests with only the existing
      Diffusers deprecation warning. The focused client graph/action matrix
      passed 103/103, `npm run check` passed, the selected-pipeline field-update
      and tamper browser contract passed 1/1, and the complete mocked Studio
      suite passed 90/90 in 283 seconds. The production bundle was
      522359/523264 gzip bytes, 777 bytes below the stricter 523136-byte safety
      target. Ruff 0.12.7 E9/F, `py_compile`, `uv pip check` (78 packages),
      preflight, formatting, lint, type, build, bundle, and diff checks passed.
      This is static, unit, contract, mocked-browser, build, and local preflight
      evidence only; no model download, video execution, generated media, or
      live workload qualification occurred.
    - [x] Expert CUDA resource-policy overlay. Backend commit `b1f514f`
      adds a bounded schema-v1 policy to each exact Qwen execution profile for
      blocked CUDA dtypes, the recommended replacement dtype, projected
      offloaded/resident VRAM, and per-quantization resident overrides. Other
      profiles omit the policy rather than publishing a nullable or inferred
      contract. Client commit `0259624` strictly parses the bounded policy and
      consumes it only from the unique execution profile named by the selected
      exact specification; the Qwen-family and 10/24/80 GiB readiness branches
      are removed. A deliberately different 12 GiB unit receipt and a
      float16-blocking mocked-browser receipt prove that the backend profile,
      not a retained client constant, controls the result. The focused backend
      matrix passed 131 tests with 367 subtests, and the complete backend gate
      passed 1,167 tests with 4 skips and 1,932 subtests with only the existing
      Diffusers deprecation warning. The focused client graph/request/readiness
      matrix passed 137/137, `npm run check` passed, the exact policy browser
      contract passed 1/1, and the complete mocked Studio suite passed 91/91.
      The production bundle was 522840/523264 gzip bytes, 296 bytes below the
      stricter 523136-byte safety target. Ruff 0.12.7 E9/F, `py_compile`, `uv
      pip check` (78 packages), preflight, formatting, lint, type, build,
      bundle, and diff checks passed. This is static, unit, contract,
      mocked-browser, build, and local preflight evidence only; no model
      download, inference, generated media, or live workload qualification
      occurred.
    - [x] Expert quantization resource-policy overlay. Backend commit
      `d3125dd` adds one bounded schema-v1 policy to the six exact Qwen
      execution profiles, declaring the Expert quantization/offload modes,
      generic Modular quantization node, reviewed component/subfolder, BnB
      quant type, compute dtype, and double-quant setting. Client commit
      `5e8a1e7` strictly parses the policy and resolves it only through the
      unique execution profile named by the selected exact specification.
      Direct specifications derive required loader fields from their exact
      bindings; Modular specifications create, populate, connect, and seal the
      declared generic quantization node without a model-family or pipeline-name
      fallback. Missing fields, node definitions, or incompatible registry
      options fail closed. The focused backend profile suite passed 9 tests
      with 81 subtests, the adjacent backend matrix passed 98 tests with 114
      subtests, and the complete backend gate passed 1,168 tests with 4 skips
      and 1,941 subtests with only the existing Diffusers deprecation warning.
      The focused client graph/request/readiness matrix passed 137/137,
      `npm run check` passed, the exact policy/topology and prior-regression
      browser matrix passed 4/4 plus the preserved expanded-node contract 1/1,
      and the complete mocked Studio suite passed 91/91 in 288.5 seconds. The
      production bundle was 523099/523264 gzip bytes, 37 bytes below the
      stricter 523136-byte safety target. Ruff 0.12.7 E9/F, `py_compile`, `uv
      pip check` (78 packages), preflight, formatting, lint, type, build,
      bundle, and diff checks passed. This is static, unit, contract,
      mocked-browser, build, and local preflight evidence only; no model
      download, inference, generated media, or live workload qualification
      occurred.
    - [x] Expert MPS resource-policy overlay. Backend commit `f2ec7ac` adds a
      bounded schema-v1 advisory to each exact reviewed Qwen, video, and
      Z-Image execution profile that has an Apple MPS qualification status and
      fallback action. Other profiles omit the policy. Client commit `06ca70f`
      strictly parses the policy and resolves it only through the execution
      profile selected by the exact specification; the previous Qwen-family,
      Z-Image-family, and video-output readiness branches are removed. The
      advisory remains Expert-only and non-blocking. The focused backend
      profile/runtime matrix passed 100 tests with 140 subtests, and the
      complete backend gate passed 1,170 tests with 4 skips and 1,967 subtests
      with only the existing Diffusers deprecation warning. The focused client
      graph/request/readiness matrix passed 137/137, `npm run check` passed,
      the exact MPS browser contract passed 1/1, and the complete mocked Studio
      suite passed 92/92 in 310.7 seconds. The production bundle was
      523116/523264 gzip bytes, 20 bytes below the stricter 523136-byte safety
      target. Ruff 0.12.7 E9/F, `py_compile`, `uv pip check` (78 packages),
      preflight, formatting, lint, type, build, bundle, and diff checks passed.
      This is static, unit, contract, mocked-browser, build, and local preflight
      evidence only; no model download, inference, generated media, Apple
      Silicon workload, or live qualification occurred.
    - [x] Generic image and Modular field-contract overlay. Backend commit
      `98f3841` gives every reviewed generic Diffusers image pipeline/mode an
      exact field-visibility overlay and makes the connected Generate, Edit,
      Inpaint/Outpaint, and Control Generate node validate and apply the whole
      selected loader contract. The same audit proves Encode Prompt, Denoise,
      Image Encode, Decode Latents, and Image Embeddings rebuild their generic
      fields from the selected Modular pipeline's registry metadata without a
      model-name switch. Client commit `c88e685` adds a mocked-browser contract
      that switches one live generic Edit node across Flux Redux
      multi-reference, SDXL img2img, and Qwen Image Edit Plus multi-reference
      selections and observes the fields changing in place. The focused image
      matrix passed 76 tests with 2 skips and 201 subtests, the adjacent
      image/Modular/profile matrix passed 134 tests with 2 skips and 504
      subtests, and the complete backend gate passed 1,174 tests with 4 skips
      and 1,996 subtests with only the existing Diffusers deprecation warning.
      `npm run check` passed, the exact image switching browser contract passed
      1/1, and the complete mocked Studio suite passed 93/93 in 308.7 seconds.
      The production bundle remained 523116/523264 gzip bytes, 20 bytes below
      the stricter 523136-byte safety target. Ruff 0.12.7 formatting and E9/F,
      `py_compile`, `uv pip check` (78 packages), preflight, client lint/type/
      build/bundle, and diff checks passed. This is static, unit, contract,
      mocked-browser, build, and local preflight evidence only; no model
      download, inference, generated media, or live qualification occurred.
    - [x] Exact image-path and Expert quantization-choice cleanup. Client
      commit `a63d882` removes the remaining image-facade registry, Flux-family,
      Qwen/Z-Image mode, and pipeline-class fallbacks from managed graph
      construction; only the exact selected execution profile or an already
      bound managed role may select the direct image facade and loader class.
      Backend commit `fd514f7` publishes a bounded, unique list of reviewed
      Expert quantization choices on each exact Qwen and Flux execution profile,
      while profiles without reviewed choices omit the field. Client commit
      `5abfab9` strictly parses that list and renders the Expert selector only
      from the unique profile selected by the exact specification. The same
      client checkpoint preserves the bound execution-spec receipt during tab
      hydration, so restored controlled graphs validate the same specification
      instead of losing their authority. The focused backend matrix passed 15
      tests with 151 subtests, and the complete backend gate passed 1,176 tests
      with 4 skips and 2,021 subtests with only the existing Diffusers
      deprecation warning. The focused client contract passed 27/27, the
      complete `npm run check` passed, and the final mocked Studio suite passed
      94/94 in 290.8 seconds. The production bundle was 522877/523264 gzip
      bytes, 259 bytes below the stricter 523136-byte safety target. Ruff E9/F,
      `uv pip check` (78 packages), preflight, formatting, lint, type, build,
      bundle, and diff checks passed. This is static, unit, contract, mocked-
      browser, build, and local preflight evidence only; no model download,
      inference, generated media, or live qualification occurred.
    - [x] Exact installed-model loader insertion. Client commit `9cec2db`
      removes the remaining Qwen/family loader-choice branches from the model
      library insertion path. A known catalog model now derives its loader
      module, action, and `model_type` or `pipeline_class` identity from the
      authoritative backend execution profiles; an authoritative catalog that
      has no matching execution profile fails closed instead of guessing a
      facade. Unknown/manual artifacts retain the explicitly editable generic
      image/audio/video and Modular fallback. The focused browser regression
      proved Qwen Image inserts `DiffusersImage.LoadPipeline` with
      `QwenImagePipeline`, and that removing its authoritative execution
      profile leaves the canvas unchanged with a bounded error. The complete
      `npm run check` passed, the complete mocked Studio suite passed 95/95,
      and the production bundle was 522977/523264 gzip bytes, 159 bytes below
      the stricter 523136-byte safety target. Type, build, bundle, and diff
      checks passed. This is static, unit, contract, and mocked-browser evidence
      only; no model download, inference, generated media, or live qualification
      occurred.
    - [x] Exact Expert quantization retention on model changes. Client commit
      `d3ad700` removes the local `Qwen Image` family exception from form
      mutation. The model selector now retains an Expert quantization choice
      only when the exact target model/mode execution profile declares that
      choice; missing, invalid, or non-declaring profiles reset to `none`.
      The focused browser regression proved `bnb_4bit` survives Qwen-to-FLUX
      switching and is removed when switching to Z-Image. The complete
      `npm run check` passed, the complete mocked Studio suite passed 96/96 in
      290.8 seconds, and the production bundle was 522954/523264 gzip bytes,
      182 bytes below the stricter 523136-byte safety target. Formatting, lint,
      type, unit/contract, build, bundle, browser, and diff checks passed. No
      model download, inference, generated media, or live qualification
      occurred.
    - [x] Declarative low-memory preset selection. Client commit `229b5d1`
      removes the Qwen- and Wan-family branches from both Studio low-memory
      entry points. The selected model profile now owns the form patch for
      dimensions, frame count, steps, dtype, quantization reset, and offload;
      the existing distinct Wan 2.2 and LTX values are therefore no longer
      overwritten by the legacy Wan VACE preset. The focused contract covers
      Qwen Image, Wan VACE, Wan 2.2 I2V, and LTX. The complete `npm run check`
      passed, the complete mocked Studio suite passed 96/96 in 295 seconds,
      and the production bundle was 522708/523264 gzip bytes, 428 bytes below
      the stricter 523136-byte safety target. Formatting, lint, type, unit/
      contract, build, bundle, browser, and diff checks passed. This is a
      declarative client-profile cleanup; moving all low-memory dimensions and
      frame limits into backend execution specifications remains part of the
      parent metadata-ownership audit. No model download, inference, generated
      media, or live qualification occurred.
    - [x] Generic Modular readiness identity. Client commit `a32b37a`
      removes the last `Qwen Image` family check from managed Run readiness.
      An existing/restored graph is classified from its generic managed
      ModelsLoader, prompt, and denoise roles; before a graph exists, readiness
      uses the exact backend execution profile's `modular-diffusers` path.
      The focused 43/43 graph-visual matrix includes the legacy restore path,
      the complete `npm run check` passed, and the complete mocked Studio suite
      passed 96/96 in 291.9 seconds. The production bundle was
      522707/523264 gzip bytes, 429 bytes below the stricter 523136-byte safety
      target. Formatting, lint, type, unit/contract, build, bundle, browser,
      and diff checks passed. No model download, inference, generated media,
      or live qualification occurred.
    - [x] Exact restored-quantization admission. Client commit `0c3a4c5`
      removes the Qwen/FLUX family filter from persisted Studio form coercion.
      Persistence now retains only the existing bounded quantization enum and
      Expert readiness permits a non-`none` choice only when the exact backend
      execution profile declares it; stale or unsupported restored choices
      block with a bounded corrective issue instead of being guessed or
      silently reinterpreted by the client. The focused 68/68 contract matrix,
      complete `npm run check`, and complete 96/96 mocked Studio suite passed;
      the browser run completed in 293.7 seconds. The production bundle was
      522741/523264 gzip bytes, 395 bytes below the stricter 523136-byte safety
      target. Formatting, lint, type, unit/contract, build, bundle, browser,
      and diff checks passed. No model download, inference, generated media,
      or live qualification occurred.
    - [x] Final residual client audit and graph-construction cleanup. Client
      commit `4afc515` removes the remaining model- and pipeline-name branches
      from fallback role selection and form-to-graph value synchronization.
      Exact execution specifications remain authoritative for current managed
      profiles; pre-specification registered facades retain their generic
      dynamic-definition fallback and consume the selected Auto candidate or
      backend execution profile without inventing a client pipeline class.
      Remaining model comparisons are identity lookup/filtering, explicit
      versioned template recipe data, or imported/manual Expert graph inference,
      not managed frontend execution routing. Source contracts passed 43/43,
      the three affected registered-facade/dynamic-definition/MPS browser cases
      passed together, the complete `npm run check` passed, and the final mocked
      Studio suite passed 96/96 in 294.7 seconds. The production bundle was
      522653/523264 gzip bytes, 483 bytes below the stricter 523136-byte safety
      target. One preceding full browser run hit the known media-format popover
      close race at 95/96; that unrelated test passed alone in 3.3 seconds and
      the unchanged complete rerun passed 96/96. No model download, inference,
      generated media, or live qualification occurred.
- [x] **P0.4 Proof receipts and current mismatch cleanup**
  - Backend: bind history to profile/schema version, graph and loader topology,
    auxiliary repositories, adapters, LoRAs, ControlNets, runtime profile, and
    artifact revisions.
  - Client: invalidate stale proof after any bound field changes and label proof
    levels accurately.
  - Tests: Z-Image loader identity, Qwen mode mappings, Wan profile closure,
    Flux Kontext multi-reference, receipt invalidation, and all checked-in graphs.
  - [x] Auto schema/profile history receipt binding. Backend commit `bf0af6b`
    corrects the schema-v2 planner so every declared candidate publishes its
    exact `executionProfileId` and owning `autoResourceSchemaVersion`; runtime
    admission now requires both values to match the current backend contract.
    History schema v3 at those commits binds local success/failure evidence to those identities,
    so an older history schema, replaced execution profile, or changed planner
    schema cannot promote a current candidate to `live_proven`. Client commit
    `4cad1b2` preserves the typed receipt and adds the positive response-boundary
    contract. This also closes the cross-repository P0.2 regression in which
    the frozen client correctly rejected real backend candidates because their
    required profile ID was absent while complete mocked fixtures supplied it.
    The focused backend matrix passed 169 tests with 298 subtests; the complete
    backend gate passed 1,138 tests with 4 skips, the existing Diffusers
    deprecation warning, and 1,772 subtests. The focused client request suite
    passed 12/12, the complete `npm run check` passed, and the complete mocked
    Studio browser suite passed 87/87 in 221.6 seconds. The production bundle
    remained 523109/523264 gzip bytes, 27 bytes below the stricter 523136-byte
    target. Ruff E9/F, `py_compile`, `uv pip check` (78 packages), preflight,
    type, formatting, lint, and diff checks passed. All 26 generated client
    files remained byte-identical and 317 Gallery files were preserved. A fresh
    backend returned ready health and HTTP 200 for `/`; a real Flux plan
    returned two schema-v2 candidates, both bound to `flux-schnell:direct` and
    candidate schema 2. The served entry SHA-256 was
    `6c1c05f0b199746275d7ef008078a1f0ebea2bfce149b531e7a533f1637e271d`.
    Its exact five-process tree was stopped and ports 8088/8089 were free. This
    is static/unit/contract/mocked-browser/local-HTTP evidence only; no model
    download, inference, generated media, or live workload qualification ran.
  - [x] Auto optional-runtime receipt binding. Backend commit `f0ccd13`
    advances local history to schema v4 and binds success/failure evidence to
    the exact optional-runtime profile list plus the immutable requirement
    schema, delivery mode, `requiredNow` flag, and execution-profile IDs.
    Runtime admission independently recomputes that contract from the current
    backend execution profile and rejects a selected/list receipt that is
    self-consistent but stale. Transient package state and explanatory reason
    remain outside the history identity; graph-derived optional-runtime
    admission remains authoritative immediately before execution. The focused
    backend compatibility matrix passed 231 tests with 490 subtests, and the
    complete backend gate passed 1,138 tests with 4 skips, the existing
    Diffusers deprecation warning, and 1,774 subtests. The client receipt/run
    contracts passed 78/78, complete `npm run check` passed, and the exact
    optional-runtime, atomic Auto submission, and wrong-loader mocked-browser
    cases passed 3/3. The unchanged production client remained
    523109/523264 gzip bytes and all 26 generated files matched the backend
    mirror byte-for-byte. Ruff E9/F, `py_compile`, `uv pip check` (78
    packages), preflight, and diff checks passed. This is static, unit,
    contract, and mocked-browser evidence only; it does not qualify an overlay,
    install packages, download a model, or execute a workload.
  - [x] Specification-owned Auto graph receipt binding. Backend commit
    `3a0b355` and client commit `0131ea7` advance local history to schema v5
    and bind each of the 32 currently specification-owned model/mode pairs to
    the exact Studio execution-spec schema version, ID, content hash, and
    execution-profile ID. The client requires that candidate contract to match
    the active managed binding before apply or Run. Runtime admission
    independently recomputes the current backend contract, requires the graph's
    role-to-node receipt, and revalidates the exact reviewed nodes, typed edges,
    and form bindings before execution. Missing, extra, malformed, or stale
    receipts fail closed, and older history cannot promote a changed graph
    contract to `live_proven`. The seven exact profile pairs that do not yet
    have a backend Studio execution specification remain explicitly outside
    this claim rather than receiving an inferred topology receipt. The focused
    backend matrix passed 231 tests with 493 subtests; the complete backend gate
    passed 1,138 tests with 4 skips, the existing Diffusers deprecation warning,
    and 1,777 subtests. Focused client contracts passed 78/78, the complete
    `npm run check` passed, and the complete mocked Studio browser suite passed
    87/87 in 229.1 seconds. The production bundle was 523120/523264 gzip bytes,
    16 bytes below the stricter 523136-byte target; all 26 generated files
    matched the backend mirror byte-for-byte and all 317 Gallery files were
    preserved. Ruff E9/F, `py_compile`, `uv pip check` (78 packages), preflight,
    type, formatting, lint, and diff checks passed. A fresh local HTTP plan
    returned `flux-schnell:text-to-image:v1`, content hash
    `studio-spec-v1-9cd1abb5`, and profile `flux-schnell:direct`; the served
    1,034,853-byte entry matched SHA-256
    `c2baccd69a6c5302c4063d3b124e107d28f20c8a9a46a9355574d1b539a527e2`.
    Its exact process tree was stopped and ports 8088/8089 were free. This is
    static, unit, contract, mocked-browser, build, and local-HTTP evidence only;
    no model download, inference, generated media, or live workload
    qualification occurred.
  - [x] Auto auxiliary-artifact receipt binding. Backend/bundle commit
    `e2a1bf2` and client commit `453da03` bind the two currently declared
    auxiliary or internally loaded model dependencies to exact bounded
    `id`/`kind`/repository/immutable-revision receipts: Qwen ControlNet Union
    for `QwenImageModularPipeline:control_image`, and FLUX.1-dev for
    `FluxReduxPipeline:edit_image`. Schema-v2 candidates, selected/list and
    retry identity, exact-pair public requirements, runtime hints, admission,
    and local history schema v6 all carry or independently recompute the same
    receipt; missing, extra, stale, malformed, or top-level/candidate-mismatched
    dependencies fail closed. Capability metadata now publishes the same
    immutable requirements. The focused backend matrix passed 139 tests with
    317 subtests; the complete backend gate passed 1,141 tests with 4 skips,
    the existing Diffusers deprecation warning, and 1,781 subtests. Focused
    client contracts passed 79/79, the complete `npm run check` passed, and the
    complete mocked Studio browser suite passed 87/87 in 225.1 seconds. The
    production bundle was 523132/523264 gzip bytes, four bytes below the
    stricter 523136-byte target; all 26 generated files matched the backend
    mirror byte-for-byte and all 317 Gallery files were preserved. Ruff 0.12.7
    E9/F, `py_compile`, `uv pip check` (78 packages), preflight, formatting,
    lint, type, and diff checks passed. A fresh local server returned ready
    health and HTTP 200 for `/`; its real Qwen Control Auto plan and public
    capability response both returned the exact ControlNet Union commit, while
    the capability response also returned the exact FLUX.1-dev Redux commit.
    The verified process tree stopped and port 8088 was free. This is static,
    unit, contract, mocked-browser, build, and local-HTTP evidence only; no
    dependency download, model inference, generated media, or live workload
    qualification occurred. Executable controlled-LoRA receipts are closed by
    the following bounded slice; future auxiliary dependencies remain pending.
  - [x] Executable controlled-LoRA Auto receipt binding. Backend commit
    `5cb785d` advances local history to schema v7 and derives an ordered receipt
    from the submitted executable graph immediately before Auto admission. The
    worker recognizes the reviewed Modular, direct-image, and direct-audio LoRA
    node contracts, resolves their Hub or local Safetensors through the existing
    exact descriptor boundary, rehashes the bytes, excludes disconnected/no-op
    adapter nodes, and binds module/action, safe artifact identity, adapter
    name, scale, scheduler, replacement policy, and descriptor digest. Submitted
    `controlledArtifacts` claims are discarded; the server copies only its own
    derived receipt into the selected/list candidate identity, resident-cache
    signature, and success/failure history. A base-only `live_proven` result is
    downgraded for a nonempty adapter set unless exact current schema-v7 history
    exists, while passed or independently safe candidates retain their proof.
    Local roots never enter the public receipt or bounded error envelope, and
    every loader still revalidates its exact bytes immediately before mutation.
    Focused backend tests passed 143 tests with 327 subtests; the exact final
    backend tree passed 1,147 tests with 4 skips, the existing Diffusers
    deprecation warning, and 1,781 subtests. Ruff 0.12.7 E9/F, `py_compile`,
    `uv pip check` (78 packages), preflight, and diff checks passed. The
    unchanged client passed complete `npm run check`; its bundle remained
    523132/523264 gzip bytes, and the schema-v3 controlled-family mocked-browser
    replay passed 1/1. This is static, unit, contract, build, and mocked-browser
    evidence only: no adapter/model download, model execution, generated media,
    or live workload qualification occurred. The current non-LoRA controlled
    artifact set and client proof labeling are closed by the following bounded
    slice; future artifact families require their own reviewed receipts.
  - [x] Current controlled-workflow artifact receipt closure. Backend commit
    `31cbc47` advances Auto history to schema v8 and derives every current
    executable controlled-artifact receipt immediately before admission. It
    retains the existing exact LoRA receipts, resolves and rehashes Spandrel
    upscalers through their pinned Hub snapshot or redacted local-file identity,
    and binds soundtrack/lyric auxiliary Diffusers pipelines to their exact
    repository, immutable revision, loader class, and descriptor digest. The
    selected primary Auto loader remains owned by its existing artifact receipt;
    disconnected nodes are excluded, submitted receipt claims are discarded,
    and the Spandrel loader rechecks declared revision, size, and SHA-256 before
    loading. Client commit `54a610a` preserves exact revision/hash/size metadata
    through each current controlled builder and labels base-only proof accurately
    for LoRA, upscaler, soundtrack, and lyric-video contracts. Focused backend
    tests passed 150 tests with 327 subtests; the exact backend tree passed 1,183
    tests with 4 skips, 2,021 subtests, and only the existing Diffusers
    deprecation warning. Ruff 0.12.7 E9/F, `uv pip check` (78 packages),
    preflight, and diff checks passed. Focused client graph/template contracts
    passed 112/112, the exact artifact browser cases passed 2/2, complete
    `npm run check` passed, and the complete mocked Studio suite passed 97/97.
    The production bundle was 523069/523264 gzip bytes, 195 bytes below the hard
    cap and 67 bytes below the stricter 523136-byte safety target. This is static,
    unit, contract, build, and mocked-browser evidence only: no artifact/model
    download, model execution, generated media, or live workload qualification
    occurred.
  - [x] Z-Image Auto execution-spec closure. Backend commit `a4efd6c` adds
    `z-image:text-to-image:v1` as the thirty-third backend-owned Studio
    execution specification and binds the existing `z-image:auto` profile to
    the exact five-node direct-image graph: `modules.DiffusersImage.LoadPipeline`,
    `ZImagePipeline`, the reviewed `Tongyi-MAI/Z-Image-Turbo` artifact, and the
    generic quantization/recipe/generate/preview route. The public capability,
    selected Auto candidate, managed graph, runtime receipt, and backend
    admission now share content hash `studio-spec-v1-0d3c1205`; wrong node
    identity or a missing/stale receipt fails closed. Client commit `77ceab9`
    adds the exact schema-v2 capability fixture and proves the existing generic
    materializer seals the Z-Image receipt and loader class without changing
    production client source. The exact final backend tree passed 1,148 tests
    with 4 skips, 1,781 subtests, and only the existing Diffusers deprecation
    warning. The focused backend contract passed 80 tests with 287 subtests;
    Ruff 0.12.7 E9/F, `py_compile`, `uv pip check` (78 packages), preflight,
    and diff checks passed. Complete client `npm run check` passed with the
    unchanged 523132/523264-byte gzip bundle, exactly four bytes below the
    stricter 523136-byte target, and the exact mocked Studio execution-spec
    browser contract passed 1/1. Six exact execution-profile pairs remain
    without backend-owned Studio specifications. This is static, unit,
    contract, build, and mocked-browser evidence only; no model download,
    inference, generated media, or live workload qualification occurred.
  - [x] Qwen Image text-to-image execution-spec closure. Backend commit
    `6e40bab` adds `qwen-image-2512:text-to-image:v1` as the thirty-fourth
    backend-owned Studio specification. It binds the existing
    `qwen-image:t2i-direct` profile, `modules.DiffusersImage.LoadPipeline`,
    `QwenImagePipeline`, official `Qwen/Qwen-Image-2512` default, reviewed
    prequantized fallback, and the generic quantization/recipe/generate/preview
    topology to content hash `studio-spec-v1-f53ab380`. Every schema-v2 Auto
    candidate carries that exact contract, and the backend independently
    requires the matching managed graph receipt before execution. Client commit
    `531d4b9` extends only the mocked capability fixture and verifies the
    selected candidate, graph binding, shared node IDs, and direct loader class;
    production client source and bundle are unchanged. The focused backend
    matrix passed 86 tests with 329 subtests; the complete backend gate passed
    1,149 tests with 4 skips, 1,781 subtests, and only the existing Diffusers
    deprecation warning. Ruff 0.12.7 E9/F, `py_compile`, `uv pip check` (78
    packages), preflight, and diff checks passed. Complete client `npm run
    check` passed with the unchanged 523132/523264-byte gzip bundle, four bytes
    below the stricter 523136-byte target, and the exact mocked Studio
    execution-spec browser contract passed 1/1. Five exact execution-profile
    pairs remain without backend-owned Studio specifications. This is static,
    unit, contract, build, and mocked-browser evidence only; no model download,
    inference, generated media, or live workload qualification occurred.
  - [x] Qwen Image Edit Modular execution-spec closure. Backend commit
    `4596728` adds `qwen-image-edit:edit-image:v1` as the thirty-fifth
    backend-owned Studio specification and binds the existing
    `qwen-edit:modular` profile to the exact seven-role, thirteen-edge Modular
    graph and fifteen form bindings. Capability publication and Auto candidates
    carry content hash `studio-spec-v1-ae6a6ce8`; backend validation resolves
    the reviewed pipeline's authoritative dynamic node definitions before
    publishing the receipt, and execution admission independently rechecks the
    exact executable node identities, edges, and bindings. Client commit
    `e8aab4e` extends the bounded role vocabulary and generic specification
    materializer so a Modular receipt waits for the backend-issued fields and
    typed route-state handles before sealing. It adds no model-name routing
    branch. The focused backend matrix passed 87 tests with 329 subtests; the
    exact complete backend gate passed 1,150 tests with 4 skips, 1,781 subtests,
    and only the existing Diffusers deprecation warning. Ruff 0.12.7 E9/F,
    `py_compile`, `uv pip check` (78 packages), preflight, and diff checks
    passed. Complete client `npm run check` passed with a 523107/523264-byte
    gzip bundle (157-byte hard-cap headroom and 29 bytes below the stricter
    523136-byte target). The exact mocked browser receipt passed 1/1, and the
    complete mocked Studio suite passed 87/87. Four exact execution-profile
    pairs remain without backend-owned Studio specifications. This is static,
    unit, contract, build, and mocked-browser evidence only; no model download,
    inference, generated media, or live workload qualification occurred.
  - [x] Qwen Image Edit Plus execution-spec closure. Backend commit `0e7a8f1`
    adds distinct `qwen-image-edit-plus:edit-image:v1` and
    `qwen-image-edit-plus:multi-image-reference-edit:v1` receipts as the
    thirty-sixth and thirty-seventh backend-owned Studio specifications. Both
    bind the existing `qwen-edit-plus:modular` profile and immutable
    `Qwen/Qwen-Image-Edit-2511` artifact to the same reviewed seven-role,
    thirteen-edge Modular edit graph and fifteen form bindings, with content
    hashes `studio-spec-v1-28b9f604` and `studio-spec-v1-86a68b80`. Client
    commit `57a4072` adds contract and mocked-browser coverage only; the
    production materializer already consumed both receipts without a new
    model-name branch or bundle change. The focused backend matrix passed 88
    tests with 329 subtests; the complete backend gate passed 1,151 tests with
    4 skips, 1,781 subtests, and only the existing Diffusers deprecation
    warning. Ruff 0.12.7 E9/F, `py_compile`, `uv pip check` (78 packages),
    preflight, and diff checks passed. Complete client `npm run check` passed
    with the unchanged 523107/523264-byte gzip bundle, 29 bytes below the
    stricter 523136-byte target, and the exact mocked-browser receipt passed
    1/1. Two complete 87-case mocked Studio replays each passed all exact-spec
    assertions and 86/87 overall; their different unrelated late-run failures
    (page bootstrap and workflow-tab scrolling) each passed immediately in
    isolation. Two exact execution-profile pairs remain without backend-owned
    Studio specifications. This is static, unit, contract, build, and
    mocked-browser evidence only; no model download, inference, generated
    media, or live workload qualification occurred.
  - [x] Qwen Layered execution-spec closure. Backend commit `dd594ba` adds
    `qwen-image-layered:layer-decomposition:v1` as the thirty-eighth
    backend-owned Studio specification. It binds the reviewed
    `qwen-layered:modular` profile and immutable
    `Qwen/Qwen-Image-Layered` artifact to the exact seven-role, eleven-edge
    Modular source-image/prompt/encode/denoise/decode/preview route and seventeen
    form bindings, including the pinned source resolution, layer count, and
    maximum sequence length. Client commit `ff3f9c6` proves the generic
    materializer seals content hash `studio-spec-v1-dda194f0` without a new
    model-name routing branch. The complete backend gate passed 1,152 tests
    with 4 skips and 1,781 subtests. Complete client `npm run check` passed, the
    exact mocked-browser receipt passed 1/1, and the production bundle was
    523118/523264 gzip bytes, 18 bytes below the stricter 523136-byte target.
    Ruff 0.12.7 E9/F, `py_compile`, `uv pip check` (78 packages), preflight,
    formatting, lint, type, and diff checks passed. One exact execution-profile
    pair remained at that checkpoint. This is static, unit, contract, build,
    and mocked-browser evidence only; no model download, inference, generated
    media, or live workload qualification occurred.
  - [x] Qwen Image Control execution-spec closure. Backend commit `03c358b`
    adds `qwen-image-2512:control-image:v1` as the thirty-ninth and final current
    exact-pair Studio specification. It binds the existing
    `qwen-image:modular` profile to eight reviewed roles, thirteen typed edges,
    and thirty-two form bindings, including the separate `AutoModelLoader`
    ControlNet component, exact Hub selector and immutable revision,
    control-image adapter, ControlNet bundle, and route-state chain through
    denoise and decode. Client commit `1102249` extends the bounded role and
    binding vocabulary while keeping materialization generic; the pinned model
    selector stays a Hub-selector object and the receipt content hash is
    `studio-spec-v1-2b0e0b6a`. The final backend gate passed 1,153 tests with 4
    skips and 1,781 subtests; the focused final matrix passed 148 tests with 317
    subtests. Complete client `npm run check` passed, the exact browser receipt
    passed 1/1, and the complete mocked Studio suite passed 87/87. The
    production bundle was 523129/523264 gzip bytes, seven bytes below the
    stricter 523136-byte target. Ruff 0.12.7 E9/F, `py_compile`, `uv pip check`
    (78 packages), preflight, formatting, lint, type, and diff checks passed.
    All 39 current execution-profile pairs now have backend-owned exact graph
    receipts. This is static, unit, contract, build, and mocked-browser evidence
    only; no model download, inference, generated media, or live workload
    qualification occurred.
  - [x] Controlled-workflow finalization-proof closure.
    - [x] Bounded schema-v2 upscaler repair: exclude field-level `disabled`
      from the proof because it is transient UI/signal state and is not consumed
      by graph export. A previously schema-matching, complete graph may reseal
      only after the already-modeled upscaler topology has an exact executable
      ledger and no binding divergence. The focused graph-visual suite passed
      41/41, including partial-route, field-toggle, core-schema mutation, and
      post-finalization edge-deletion cases; the complete mocked Studio suite
      passed 83/83. This is not generic controlled-extension proof.
    - [x] Schema-v3 controlled-workflow proof: begin only from a matching proof,
      perform each trusted extension as one synchronous fail-closed transaction,
      validate a strict controlled-role-to-node-key contract, preserve the
      independent exact Modular core-route check, and seal every managed
      extension node identity/schema, node execution-disabled state, parent/loop
      semantics, and actual managed edge endpoint. Restore must revalidate the
      complete hash, and abort must roll back or remain proofless. Cover LoRA
      (Modular, direct image, and direct audio), video sequence and upscaler
      composition, quality-sequence loops, soundtrack/export replacement, and
      lyric/mux workflows. These reviewed groups are now covered by the client
      proof gate; broader P0.4 receipt work and live execution remain separate.
      Evidence 2026-08-10: schema-v3 now persists a bounded controlled-contract
      declaration and seals the exact managed role/node identity, reviewed
      registry execution shape, node execution-disabled state, parent/quality-
      loop membership, and actual edge IDs/endpoints/handles. Each reviewed
      builder runs inside one synchronous begin/commit/abort boundary; partial or
      failed mutations restore the previous graph/proof/history, successful
      replacement defers cache cleanup until after commit, and sequential/reverse
      compositions retain prior receipts. Authoritative schema messages clear
      the proof before mutation and may reseal only the same topology/execution
      baseline; malformed persisted declarations remain quarantined until the
      operator explicitly detaches the invalid receipt. Field-level `disabled`
      and non-durable callbacks remain proof-neutral, while registry-declared
      type/display/input/spawn/data-source shape is revalidated independently of
      self-supplied hashes. Focused graph/template/run contracts passed 136/136,
      the full client `npm run check` passed, and the complete mocked Studio suite
      passed 85/85. The production bundle was 522753/523264 gzip bytes (511-byte
      hard-cap headroom and 383 bytes below the stricter safety target). The
      tracked packed-template codec regenerates deterministically in the normal
      unit gate; 77 runnable and 3 planning templates remained deep-identical,
      and all 360 Gallery asset paths/purpose sets remained unchanged. Independent
      graph-contract and bundle-semantic audits found no remaining blocker. This
      is static/unit/contract/mocked-browser CPU evidence only: no model execution,
      media qualification, optional package action, or live runtime cutover ran.
- [x] **P0.5 Lazy optional Hugging Face runtime installation**
  - [x] Contract/status preparation: declare one exact composite
    `transformers==5.14.1` + `peft==0.20.0` profile, bind it to every current
    Diffusers execution profile, and publish metadata-only status through Auto,
    model capabilities, and workflow listings without changing readiness.
  - [x] Staged overlay: generalize the existing package overlay, require an
    exact catalog ID/spec digest plus explicit consent, validate in a fresh
    process, serialize/cancel installs, and bind validated state to the base
    environment identity.
    - [x] Fail-closed scaffold: exact request schemas, process/worker mutation
      gates, durable jobs, cancellation/watchdog plumbing, artifact-anchored
      validation, restart/repair/rollback states, bounded public projections,
      and legacy hashless-overlay rejection are implemented and CPU-tested.
      The current candidate still rejects before lease, job, network, staging,
      or subprocess creation.
    - [x] Immutable wheel/installer preparation: bind the complete ten-wheel
      Transformers/PEFT closure to exact official PyPI filenames, URLs,
      SHA-256 values, and byte sizes for Python 3.12 on Linux, macOS, and
      Windows on x86-64 and ARM64. Share one reviewed uv `0.11.26`
      archive/executable identity with base setup, require a rehashed receipt,
      validate complete unique wheel RECORD hashes/sizes, and place the Windows
      watchdog and all descendants in a non-breakaway kill-on-close Job Object.
      Install, activation, and cutover flags remain false.
    - [x] Promotion storage preparation: bind the install lease to the original
      staging-directory identity; use held-parent exclusive, no-replace
      promotion plus handle-scoped quarantine cleanup; and persist canonical
      manifest/validation digests in a bounded promotion journal. Locked
      startup/install/activation/rollback reconciliation completes only one
      exact prepared move, acknowledges one exact promoted move, and leaves
      malformed, ambiguous, replaced, or missing states fail-closed for repair.
      Windows write-through promotion/cleanup and crash-window regressions pass;
      action flags remain false pending target-platform execution evidence.
    - [x] Windows x86-64 isolated staging proof: under temporary, process-local
      future-state flags, pinned uv installed all ten reviewed wheels
      (16,930,199 archive bytes) with copy-only cache behavior; 3,441 wheel files
      matched their archive anchor; isolated symbol/origin validation passed;
      promotion/activation succeeded; a second fresh process loaded exact
      Transformers `5.14.1` and PEFT `0.20.0` from the promoted overlay; and
      rollback returned to base with no promotion journal or path-bearing
      requirements file left behind. Source action/cutover flags remain false.
      No model artifact was downloaded or executed.
    - [x] Windows x86-64 no-weight staged workload: a second isolated run used
      the production artifact-locked install, validation, promotion,
      activation, fresh-process, and rollback path. With Hugging Face network
      access disabled, exact Transformers `5.14.1` and PEFT `0.20.0` loaded
      from the overlay; a tiny local CLIP text encoder accepted PEFT LoRA
      adapters and produced a finite `[1, 4, 16]` forward result with four
      trainable adapter parameters; Diffusers enabled its PEFT backend; and a
      fresh post-rollback process returned to base. Ten locked wheels totaling
      16,930,199 bytes were used, no model artifact was downloaded, and source
      flags stayed false. This is not clean-base, supervised server restart,
      live model/media, or non-Windows evidence.
    - [x] Windows x86-64 clean-base/staged-runtime matrix: a detached
      prospective checkout removed Transformers and PEFT from the default
      dependencies and required preflight imports, then the managed NVIDIA
      installer produced a compatible 64-package CUDA base with all ten staged
      distributions absent. Preflight was ready and registry discovery loaded
      132 nodes without loading the optional closure. From that base, the exact
      ten-wheel/16,930,199-byte overlay passed install, validation, promotion,
      activation, the finite CLIP+LoRA workload in a fresh process, and rollback
      to no active environment. The detached checkout and temporary overlay
      were removed afterward. This is not live-model, supervised-server, or
      non-Windows evidence; the real source dependency/action/cutover flags
      remain unchanged.
    - [x] Windows x86-64 supervised HTTP lifecycle: from the same prospective
      clean base, a real supervised server accepted explicit install consent,
      published bounded `installing` and `validating` progress, and retained a
      `ready` job bound to environment `runtime-1786525202-55aff828` and the
      exact profile/spec digest. Explicit activation returned
      `restarting: true`, replaced base worker `CF9j1A2gcK` with active worker
      `8ikwPhct-9`, and the new worker reported overlay status `active` plus
      Transformers `5.14.1`. Explicit rollback returned `restarting: true`,
      replaced that worker with `p49cg4_HIn`, restored process status `base`,
      and again reported Transformers absent. A real install first exposed an
      invalid keyword call across `call_soon_threadsafe`; both optional-runtime
      and legacy optimization progress dispatch now use a bound callback and
      have worker-thread regression coverage. The detached checkout, staged
      environment, server processes, and diagnostics were removed afterward;
      port 8088 was free. Source action/cutover flags remain false. This is not
      live-model, live cancellation/repair, or non-Windows evidence.
    - [x] Windows x86-64 supervised cancellation and repair: a second clean-base
      supervisor began the real locked install, exposed `installing`, accepted
      the exact job-scoped cancellation request, and reached terminal
      `cancelled` with no staging directory, no promoted environment, no worker
      replacement, and Transformers still absent. A subsequent validated
      environment was activated, then a one-byte managed-overlay drift was
      introduced in the detached qualification tree. On restart, the worker
      imported no optional package and published `repair_required` for the
      process, active environment, and profile. A new install produced a
      separately validated replacement; direct replacement activation was
      refused while the corrupt selection remained active, explicit rollback
      restarted to base, and activating the exact completed-job environment
      restarted into Transformers `5.14.1`. A final rollback returned to a new
      base worker with Transformers absent. Setup already prioritizes that exact
      completed-job environment receipt over catalog inference. All temporary
      processes and managed state were removed. This is not live model/media or
      non-Windows evidence.
    - [x] Windows x86-64 guarded live-model execution: a detached clean-base
      checkout enabled the complete future qualified/action/cutover contract
      only in that qualification tree. Model Manager downloaded the
      Apache-2.0, safetensors-only, no-custom-code
      `optimum-intel-internal-testing/tiny-random-qwen-image` snapshot at exact
      commit `ef73a0df0cb8ccfa00cc178ec528c6e681791a10`; validation observed
      17 complete files and 41,663,402 completed bytes and confirmed
      `QwenImagePipeline`. A fresh supervised worker activated the exact
      ten-wheel composite overlay and ran the existing generic
      `DiffusersImage.LoadPipeline -> Generate -> Image.Save` path on an RTX
      4080 at 64 by 64, one step, seed 123. Task `W44WcoUEma-X` completed in
      1.25 seconds and wrote a non-uniform RGB PNG with pixel digest
      `sha256:8aef57fdb4aa58e5d2dcacb04b731893d3883f30554e1004e9c49b863a083e6f`;
      its runtime receipt reported Diffusers `0.40.0.dev0`, Transformers
      `5.14.1`, Torch `2.8.0+cu128`, and CUDA execution. After explicit
      rollback, the same exact Qwen loader was rejected before queueing with
      HTTP 409 `optional_runtime_staged`, `requiredNow: true`, and no current
      task. The run exposed that custom immutable loader pins could not be
      passed through Model Manager; `/hf_download` now accepts only an exact
      lowercase 40-character optional `revision`, binds concurrent joins to
      that revision plus file selection, and forwards it to the app-owned Hub
      snapshot operation. The qualification artifact, output, overlay, and
      processes were removed afterward. Production dependency, action,
      qualification, and cutover declarations remain unchanged. This is not
      non-Windows evidence and does not itself authorize the atomic base
      dependency cutover.
    - [x] Portable non-Windows qualification preparation: add an explicit-
      consent, path-redacted qualification command that refuses a non-Python-
      3.12 host, an unverified managed uv executable, or any base interpreter
      containing one of the ten staged distributions. In a disposable managed
      root it projects the future qualified profile only in memory, uses the
      production locked install/validation/promotion/activation path, runs the
      offline CLIP+LoRA workload in a fresh child, rolls back, and verifies a
      second fresh clean-base child. Focused contract tests cover dormant source
      flags, consent-before-preflight, exact artifact selection, forged uv
      rejection, and bounded no-overwrite evidence. This prepares a reproducible
      Linux/macOS handoff; it is not platform evidence until executed there and
      does not replace supervised HTTP restart/cancel/repair or live model/media
      qualification. No local macOS host is available; macOS must remain pending
      until the manual reviewed hosted workflow or a contributor-controlled Mac
      produces the same reviewed evidence. Do not enable global action/cutover
      flags in the interim. A Windows-and-Linux-only release would first require
      an independently reviewed platform-scoped delivery contract that keeps
      macOS base-delivered.
    - [x] Linux x86-64 clean-base/staged-runtime and supervised lifecycle: a
      detached prospective checkout at `b30c6b1` removed only Transformers and
      PEFT from project dependencies. The managed Python `3.12.13` CPU installer
      produced a compatible 61-package base with all ten staged distributions
      absent; preflight was ready and discovery loaded 133 nodes from 20 modules
      without Transformers. The portable qualifier installed and validated the
      exact 10-wheel/17,457,395-byte Linux plan, ran the offline finite
      `[1, 4, 16]` CLIP+LoRA workload with Transformers `5.14.1` and PEFT
      `0.20.0` from overlay-bound origins, and rolled back to a fresh clean-base
      process. Its 1,518-byte path-free evidence digest was
      `sha256:a90c3c84826a55af615b62eda9b8ea83c356691330eabba52e54ad3b0b790cde`.
      A qualification-only supervised server then proved explicit consent,
      `installing`/`validating` progress, job cancellation, reinstall,
      activation with worker replacement, same-size one-byte drift detection,
      `repair_required` without optional imports, direct corrupt-selection
      replacement refusal, rollback, exact completed-job replacement
      activation, and final clean-base restart. No model/media asset or Gallery
      content was downloaded. All qualification state was removed and both
      ports were free. This is Linux CPU package/no-weight and HTTP lifecycle
      evidence, not AMD GPU, live-model/media, macOS, or production-cutover
      evidence; source flags remain dormant.
    - [ ] macOS ARM64 executable qualification: the manual-only
      `.github/workflows/qualify-optional-runtime-macos.yml` proposal asserts the
      hosted architecture on the explicit `macos-15` ARM64 standard runner,
      retains the prospective dependency diff, requires ready preflight, runs
      the consented qualifier, and uploads bounded evidence for review. It has
      not run and makes no macOS success claim. The
      [official GitHub-hosted runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
      was rechecked on 2026-08-14 and still identifies `macos-15` as ARM64; the
      workflow's independent `uname -m` assertion remains the fail-closed
      runtime authority.
    - [x] Enabled-target executable qualification: Windows and Linux x86-64
      have reviewed wheel/installer, clean-base/staged workload, fresh-process,
      restart, and rollback evidence. The four pending target rows remain
      base-delivered and non-actionable until separately qualified.
  - [x] First-use execution guard and client install/activation/restart flow.
    - [x] Cutover-dormant guard/status scaffold: exact backend execution
      profiles own a versioned seven-field requirement; every current profile
      remains `base`/`requiredNow: false`; graph and field-action admission are
      rechecked at the worker and pre-import boundaries; and the client exposes
      read-only Setup status while blocking only an authoritative
      `optional_overlay` requirement. The unqualified current catalog exposes no
      install or activation control and sends no package mutation request.
      Evidence 2026-08-10: the focused backend guard/status matrix passed 371
      tests with 4 skips and 744 subtests in 22.66 seconds; an independent replay
      passed the same matrix in 22.53 seconds. The complete backend gate passed
      1086 tests with 4 skips and 1565 subtests in 42.02 seconds, with only the
      existing Diffusers `torch_dtype` deprecation warning. Scoped `py_compile`,
      Ruff E9/F, `uv pip check` (78 compatible packages), preflight, port, and
      diff checks passed. Client optional-runtime contracts passed 111/111 and
      graph-mutation contracts passed 24/24. The final warning-clean
      `npm run check` passed with a 523072/523264-byte gzip bundle (192-byte
      headroom); the complete mocked Studio suite passed 83/83, including the
      exact GET-only optional-runtime Setup contract. That contract proves no
      install, activate, or rollback control/request is exposed while actions
      remain unavailable, and that loading, restart, repair, and qualified-active
      status are rendered without stale-active authorization. Independent audits
      signed only this cutover-dormant, base-neutral scaffold. These are
      static/unit/contract/mocked-browser CPU results: no package action,
      dependency cutover, network model download, model execution, media, or
      live overlay qualification occurred.
      All 26 generated client files were verified byte-identical in the backend
      mirror while preserving `web/template-gallery`; the served `index.js` was
      1,005,768 bytes with SHA-256
      `3f71f642055631774a51f1e1e69c5fc5585abe82814016ebd691b1fe69517d6b`.
      A fresh worker returned HTTP 200 for `/health`, `/`, and
      `/assets/index.js`; the worker and temporary logs were removed and port
      8088 was free afterward.
    - [x] Cutover-dormant client controls: Setup parses the bounded backend
      environment/job contracts and exposes generic install or repair, progress,
      cancellation, activation, and rollback controls only when the exact
      profile is backend-qualified, cutover-ready, and action-enabled. Install,
      activation, and rollback each require explicit consent; polling is bounded
      to the returned job/profile/spec identity; duplicate staged environments
      suppress activation; and the current unqualified profile remains GET-only.
      Focused optional-runtime contracts passed 33/33, the qualified mocked flow
      proved no mutation before consent plus install/progress/cancel/reinstall/
      activation/restart/rollback, and the complete mocked Studio gate passed
      98/98. The production bundle was 522936/523264 gzip bytes, 200 bytes below
      the stricter 523136-byte safety target. This is a dormant mocked-browser
      control surface, not package-action or workload qualification.
    - [x] Actionable first-use qualification: enable the controls only
      after executable overlay qualification, then exercise explicit consent,
      bounded install progress/cancellation, activation, supervised restart,
      repair, and rollback without making
      discovery, template open, Auto planning, or base-delivered execution
      depend on optional-runtime status. Do not mark this complete until
      reviewed cross-platform wheel locks and installer containment are
      qualified, backend version/symbol/origin verification succeeds in the
      activated worker, supervised restart/repair/rollback are exercised, and a
      staged-runtime workload passes.
      - [x] Linux x86-64 qualification-only projection passed real HTTP
        consent/install/progress/cancel/reinstall, fresh-worker activation,
        exact version and origin checks, drift repair, rollback, exact receipt
        reactivation, and final base restoration. Production controls remain
        dormant because macOS and the global cutover review are still open.
      - [x] Platform-scoped production cutover: commit `655baa6`, corrected by
        `1e95362`, publishes all six target rows and enables only qualified
        Linux/Windows x86-64. A clean committed Linux CPU checkout installed 61
        base packages with all ten staged distributions absent. Production
        preflight reported the exact source target qualified without an
        in-memory flag change; the 10-wheel/17,457,395-byte install, validation,
        activation, finite CLIP+LoRA child, rollback, and second clean-base
        child passed in 38.19 seconds. The bounded 1,552-byte evidence digest
        was `sha256:dd83200dbf7132e65e2e62447a1ea002e8de5ee5e22cdd454db72dc36a1b13af`.
        A fresh production worker published qualified/actionable Linux x86-64
        status, reported Transformers absent, and rejected a Z-Image graph with
        HTTP 409 `optional_runtime_missing`, `requiredNow: true`, before queueing.
        The temporary checkout, evidence, managed state, and processes were
        removed; ports 8088/8089 were free afterward. A fresh 2026-08-14 source
        revalidation of the target profile table, optional-runtime execution
        guard, qualification projection, install guidance, and Diffusers
        delivery contract passes 94 tests and 929 subtests. This is regression
        evidence over the recorded Linux/Windows cutover, not new macOS or ARM
        execution evidence.
  - [x] Atomic base cutover: remove both Transformers and PEFT only after the
    clean-base and staged-runtime qualification matrices pass. Before any
    execution profile changes to `optional_overlay`, add exact repo-aware client
    readiness for shared loader classes so local or unknown repositories match
    the backend loader-identity guard instead of being shown as runtime-ready.
    - [x] Repo-aware shared-loader readiness and Auto parity: the client now
      resolves a managed loader by exact module/action plus model/pipeline
      identity, disambiguates shared classes only with an exact Hub repository
      from each profile's default/fallback/compatible set, and preserves
      base-delivery neutrality when no candidate requires an overlay. Local,
      malformed, unknown, or ambiguous selectors fail closed as soon as any
      matching profile requires the optional runtime. Schema-v2 Auto candidates
      require one consistent artifact repository receipt; applying a reviewed
      plan may update the exact managed loader, while readiness and submission
      require the live loader to match the effective selected repository.
      Evidence 2026-08-12: focused client request/template contracts passed
      109/109; the exact wrong-repository mocked-browser regression passed 1/1;
      full `npm run check` passed; and the complete mocked Studio suite passed
      99/99 in 293.7 seconds. The production bundle remained within the fixed
      gate at 523108/523264 gzip bytes (28 bytes below the stricter 523136-byte
      safety target). All 77 runnable and 3 planning templates remained
      deep-identical after the size carve. The backend shared-loader/profile
      replay passed 41 tests and 254 subtests in 31.67 seconds, including Hub,
      compatible-repository, local, custom, malformed, and executable-path
      selection. This closes only the repository/readiness prerequisite; it is
      not clean-base, staged workload, restart, rollback, or atomic cutover
      evidence.
  - Backend: add reviewed package requirements to execution specifications;
    generalize the staged optional-runtime installer for official Hugging Face
    libraries; verify in a fresh process; support activation, restart, and
    rollback; then atomically remove both direct Transformers and PEFT
    dependencies only after the clean base profile passes. Bind a validated
    overlay to its exact runtime-spec, Python, accelerator-profile, and pinned
    Diffusers identities so stale overlays require repair instead of loading.
  - Client: when Run first needs a missing runtime, show an explicit install
    action and progress. Do not install on application setup, template open,
    node discovery, or Auto planning.
  - Tests: clean base install without Transformers or PEFT, lightweight
    discovery and preflight, missing/wrong-version/repair-required readiness,
    decline/cancel and concurrent-install exclusion, successful staged install,
    exact version/symbol/origin validation, failed validation,
    activation/restart, rollback, and an existing Diffusers workflow whose text
    encoder requires the optional composite runtime.
  - Assets: none. Hardware: CPU-only package and contract tests.
  - Audit evidence (2026-08-09): hiding Transformers while leaving PEFT present
    failed registry discovery through Diffusers `ComponentsManager` ->
    `peft.helpers`; hiding both under offline flags loaded all 20 module groups
    and 132 nodes without importing either package. Candidate pins
    `transformers==5.14.1` and `peft==0.20.0` passed local no-weight API probes,
    but remain unqualified until the managed cross-platform matrix passes.
  - Contract/status evidence (2026-08-10): the standard-library-only profile
    catalog publishes exact provenance, requirements, and canonical spec digest
    `sha256:8e1b0b6b2baa891d4551caa3cde4d59708eced0fd74c1333b68a1aab7ff924b5`.
    The executable specification names the complete overlay-owned closure:
    Transformers `5.14.1`, PEFT `0.20.0`, tokenizers `0.22.2`, Typer `0.27.1`,
    annotated-doc `0.0.5`, Rich `15.0.0`, markdown-it-py `4.2.0`, mdurl
    `0.1.2`, Pygments `2.20.0`, and shellingham `1.5.4`.
    Missing, wrong-version, unreadable, and exact-present host metadata remain
    observational; exact presence is still `present_unqualified`, with cutover,
    install, and activation unavailable. Strict discovery, Auto, capability,
    `/listgraphs`, and template-open tests prove no optional package is loaded or
    installer path invoked by those surfaces. Unknown profile IDs fail closed,
    and all three host states leave Auto selection and readiness identical.
    Focused implementation gates passed 64 tests and 219 subtests; an independent
    adjacent audit passed 109 tests and 311 subtests plus `py_compile`, Ruff E9/F,
    `uv pip check`, and diff checks. The complete backend replay passed 1009
    tests and 1400 subtests with 2 platform skips and the existing Diffusers
    deprecation warning; repository-wide Ruff E9/F, package compatibility,
    preflight, port, and diff checks also passed. The base dependency declarations
    still intentionally include both Transformers and PEFT; no installation,
    activation, dependency cutover, network access, or model execution occurred.
  - Fail-closed overlay-scaffold evidence (2026-08-10): synthetic locked-wheel
    tests bind filenames/hashes to the spec, verify the full ten-distribution
    closure, re-anchor retained caches, and reject archive replacement,
    self-consistent forged RECORDs, startup hooks, links, path aliases, and
    unsafe Windows names before optional imports. Server tests cover strict
    install/activate/rollback/cancel schemas, both graph/mutation admission
    orders, unsupervised restart blocking, durable monotonic jobs, interrupted
    job reconciliation, corrupt-state recovery, catalog bounds, and public/on-
    disk redaction. Benign subprocess tests cover cross-process lease exclusion,
    ordinary child/grandchild cancellation, and hard-worker-death watchdog
    cleanup/reacquisition. The frozen focused matrix passed 69 tests and 66
    subtests with two privilege-only symlink skips; the complete backend replay
    passed 1058 tests and 1443 subtests with four platform/privilege skips and
    the existing Diffusers `torch_dtype` deprecation warning. Scoped Ruff E9/F,
    `py_compile`, package compatibility, preflight, port, and diff checks
    passed. An independent adversarial audit signed the currently reachable
    fail-closed scaffold and explicitly did not sign enabling an overlay. This
    is static/unit/no-network CPU evidence, not a staged package install or
    model run. At that checkpoint package actions remained unavailable and the
    staged-overlay gate remained open. Legacy hashless overlays remain
    non-executable after the later platform-scoped cutover.

### Phase 0 completion gate

- [x] All focused backend tests pass.
- [x] Complete backend gate passes.
- [x] Client unit and mocked browser gates pass.
- [x] Existing supported exact pairs retain their public inputs and outputs.
- [x] Unknown or unsupported pairs cannot become Auto-ready.
- [x] No generated assets or model downloads were needed.
- [x] A clean base installation does not install Transformers or PEFT; a
  requiring workflow remains blocked until its explicit first-use composite
  runtime installation succeeds.

Clean-base test-topology revalidation (2026-08-14): backend `e29fbf4` removes
unintended top-level Transformers use from exact fixture-only tests and scopes
40 tests that construct real optional upstream objects to hosts where the
staged runtime is base-delivered or explicitly activated. The Linux clean CPU
base therefore exercises the full data-only, security, route, installer, and
fixture surface without reinstalling Transformers or PEFT; its complete gate
passed with `1625 passed, 40 skipped, 3273 subtests` and only the existing
Diffusers `torch_dtype` deprecation warning. Those skips do not constitute
macOS evidence: the physical macOS qualifier remains pending, and the skipped
contracts still run on a base-delivered or activated qualified runtime.

Hosted-runner maintenance (2026-08-14): backend `6e08028`, corrected by
`11d58a9`, moves the manual optional-runtime qualifier and the ordinary backend
CI matrix from GitHub's scheduled-for-deprecation macOS 14 image to the
explicit macOS 15 ARM64 standard runner. The correction found that the
qualifier's duplicated inline prospective-base patch still named an older
Diffusers pin and no longer applied. The workflow now creates one exact patch,
checks and applies that same byte sequence, and has a regression that actually
runs `git apply --check` against the current project. It remains manual-only
and preserves its ARM64 assertion, ready-preflight requirement, consented
execution, and bounded 14-day evidence upload. Both workflows parse as YAML;
the focused installer/runtime regression passed `50 tests, 126 subtests`, the
new executable patch regression passes, and the complete backend gate passes
`1687 tests, 40 skips, 3417 subtests` with Ruff E9/F, package compatibility,
portable preflight, shell syntax, and diff checks green. This is source/static
evidence only: the workflow has not run, no production flag changed, and the
physical macOS gate remains unchecked.

## Phase 1 — Modular foundation without large model runs

Priority: after Phase 0. Hardware: CPU and tiny fixtures. Assets: none.

### Committable segments

- [x] **P1.1 Reviewed custom Modular and DynamicBlock execution contract**
  - Backend: remove the invalid curated default and bundled graph; accept only
    `modiff_pipeline_config.json` for bounded declarative MoDiff UI metadata.
    Before enabling execution, validate every repository-supplied component
    library and class against MoDiff's reviewed official Hugging Face
    dependency contract, validate canonical upstream block/workflow metadata,
    and pin every transitive repository. Build a content-addressed private
    execution snapshot so cache or local-file mutation cannot race validation.
    Any repository Python path additionally needs a task-scoped explicit
    operator authorization that cannot be restored from imported workflow data.
  - Optional runtimes: a validated component may request Transformers or
    another separately approved Hugging Face library only through the P0.5
    first-use install/consent profile. Contract preview, template browsing,
    registry discovery, and Auto planning remain non-installing operations.
  - Client: show a neutral repository field and actionable missing-sidecar,
    unpinned-auxiliary, unapproved component, missing-runtime, immutable-snapshot,
    and trust errors. Do not mention Mellon in product UI or treat a persisted
    identity checksum as consent.
  - Tests: exact filename, no filename fallback, missing file, hostile schema,
    arbitrary installed-package dispatch, immutable main and auxiliary
    revisions, cache/local mutation and validation-to-load races, imported
    authorization replay, auth/network distinction, and no downloads or
    optional-library installation during node discovery or contract preview.
  - Status 2026-08-12: complete in backend `207d8f1`, with actionable response
    guidance corrected by `5dc7313`, and client/browser proof `c3e932c`. The
    invalid curated repository default, its bundled graph, and its catalog row
    are removed. Only the exact `modiff_pipeline_config.json` sidecar supplies
    bounded UI metadata; canonical `modular_model_index.json` supplies the
    executable pipeline/block and component contract. Hub execution requires
    an exact main commit and exact commits for every official Diffusers or
    Transformers component repository. Path-like auxiliary sources,
    repository requirements, `auto_map`, arbitrary installed-package dispatch,
    and local mutable execution fail closed. The backend revalidates identity
    immediately before a private content-addressed metadata snapshot and
    constructs only the installed reviewed Diffusers pipeline/blocks pair.
    Repository Python remains unavailable; the persisted trust field and
    contract checksum are not authorization. Preview callbacks neither inspect
    nor install the optional runtime, while both executable custom loader paths
    use the P0.5 first-use gate.
  - Evidence 2026-08-12: the focused backend contract matrix passed (`186
    passed, 628 subtests passed`); the complete backend suite passed (`1245
    passed, 3 skipped, 2107 subtests passed`), followed by the API-guidance
    focused gate (`65 passed, 1 skipped, 156 subtests passed`). Ruff `E9,F`,
    compile, and diff checks passed. Client `npm run check` passed, including
    all unit gates, build, and the bundle budget (`523239 / 523264` total gzip
    bytes); the focused mocked-browser admission test passed (`1 passed`). No
    model weights, media, repository Python, or optional-runtime install was
    downloaded or executed for this segment.
- [x] **P1.2 Generic upstream workflow discovery**
  - Backend: derive workflows, required inputs, outputs, and components from
    `available_workflows`, `get_workflow()`, block docs, and `init_pipeline()`;
    keep small reviewed overlays for MoDiff aliases and UI defaults.
  - Client: render task choices and fields from the normalized contract rather
    than pipeline-name switches.
  - Tests: Sequential, Auto, Loop, state, component reuse, schema round trip, and
    unknown workflow rejection.
  - Status 2026-08-12: complete in backend `50dafa6` and client `7c6bdbf`.
    A checked-in schema-v1 snapshot derives 11 registered pipeline contracts,
    39 workflows, required inputs, outputs, state keys, nested block kinds/docs,
    initialized execution classes, and component reuse keys from the exact
    pinned no-weight Diffusers APIs. Runtime and registry consumers validate the
    bounded snapshot without importing Diffusers. DynamicBlock publishes only
    tasks whose required inputs its reviewed sidecar can carry, rejects fields
    outside the upstream contract before construction, and filters inputs and
    outputs to the selected task. The existing generic client field action
    renders the backend-owned task choices and visibility map without a
    pipeline-name branch.
  - Evidence 2026-08-12: the focused backend matrix passed (`120 passed, 302
    subtests passed`); the complete backend suite passed (`1253 passed, 3
    skipped, 2107 subtests passed`). The pinned generator `--check`, Ruff
    `E9,F`, compile, shell syntax, package compatibility, and diff checks passed.
    Client `npm run check` passed, including build and the bundle budget
    (`523239 / 523264` total gzip bytes), and the focused mocked-browser task
    selector test passed (`1 passed`). Preflight separately reported the
    existing managed CPU environment receipt as stale for this checkout; no
    package install, model weight, network workflow discovery, or asset
    generation was performed for this segment.
- [x] **P1.3 Complete the generic guider registry**
  - Add `AdaptiveProjectedMixGuidance`, `MagnitudeAwareGuidance`, and
    `PerturbedAttentionGuidance` to the existing Guider node.
  - Test constructor parameters, required layers/components, signal updates, and
    pinned upstream exports.
  - Status 2026-08-12: complete in backend `b48355b` with generic client signal
    proof `f044594`. `AdaptiveProjectedMixGuidance`,
    `MagnitudeAwareGuidance`, and `PerturbedAttentionGuidance` are resolved from
    the exact pinned official `diffusers.guiders` namespace. This matters for
    Magnitude Aware Guidance because the pin exports it from that namespace but
    omits it from the top-level Diffusers lazy-export list; no dependency pin
    change or local implementation was introduced. The registry publishes its
    bounded `alpha` control to every reviewed guider-capable pipeline, while
    Perturbed Attention retains its exact non-empty Layers contract and all
    choices remain narrowed by the connected backend pipeline signal.
  - Evidence 2026-08-12: the focused pinned upstream matrix passed (`48 passed,
    212 subtests passed`); the complete backend suite passed (`1254 passed, 3
    skipped, 2108 subtests passed`). Ruff `E9,F`, compile, and diff checks
    passed. Client `npm run check` passed with the unchanged bundle budget
    (`523239 / 523264` total gzip bytes), and the focused generic guider signal
    browser test passed (`1 passed`). Tests constructed the three guiders
    without weights; no model, media, download, or generated asset was used.
- [x] **P1.4 Register current-pin missing Modular classes as contract-only**
  - Split into reviewable image, video, and multimodal batches.
  - Do not mark them Auto-ready or live-supported.
  - Each batch has backend class/schema tests and client experimental/Expert
    visibility tests.
  - Status 2026-08-12: complete in backend `8e44eb5` and client `5cb2998`.
    The exact pinned upstream inventory is split into six image, seven video,
    and two multimodal discovery records. All 15 publish normalized checked-in
    workflow/component schemas and appear in the generic Expert model selector,
    but remain outside the executable registry with no repository, runnable
    mode, optional-runtime, Auto, template, Gallery, or live-support claim.
    Models Loader clears their pipeline signal and rejects execution before
    artifact or pipeline-index resolution.
  - Evidence 2026-08-12: the focused backend matrix passed (`73 passed, 341
    subtests passed`) and the complete backend suite passed (`1259 passed, 3
    skipped, 2168 subtests passed`). The generator `--check`, data-only import,
    clean-base optional-import, Ruff `E9,F`, compile, package compatibility,
    shell syntax, and diff gates passed. Client `npm run check` passed with the
    unchanged bundle budget (`523239 / 523264` total gzip bytes), and the
    focused mocked-browser Expert test passed (`1 passed`). No weight, model,
    media, network artifact, or generated asset was used.

### Phase 1 completion gate

- [x] Complete backend and client gates pass.
- [x] Registry discovery imports no large model stack and downloads no weights.
- [x] Every exposed workflow is present in the pinned upstream block definition.
- [x] DynamicBlock has no Mellon filename, schema, option, or fallback.
- [x] No assets are generated.

Fresh 2026-08-14 source revalidation passed the custom identity,
DynamicBlock security, contract-only registry, pinned-upstream contract,
workflow-discovery, and workflow-truth suites (`121 passed`, `24 skipped`,
`222 subtests passed`). The run used no weights, media, network artifacts, or
generated assets.

## Phase 2 — Templates for already implemented execution paths

Priority: first user-visible expansion. Hardware: contract tests locally; live
output and assets remotely. Assets: remote Dataset only.

### Committable segments

- [x] **P2.1 Generic task-template builder and validator**
  - Backend: validate exact execution profile, loader identity, graph inputs, and
    output contract for every graph.
  - Client: generate task skeletons from generic image/audio/video contracts;
    do not clone model-specific graph builders.
  - Tests: graph round trips, required media, loader identity, stable IDs, and
    Gallery-hidden state while qualification is pending.
  - Evidence 2026-08-12: the backend now derives 39 content-addressed task
    contracts from the authoritative execution specifications and validates
    exact profile/loader identity, required media, and modality terminal output.
    Checked-in image, video, and audio graphs passed JSON round-trip validation;
    loader, output, and required-media tampering failed closed. The client
    strictly consumes the same generic contract, produces planning-only
    skeletons without another graph representation, ignores future unknown
    model types, and keeps every pending contract out of Gallery. The complete
    backend suite passed (`1263 passed, 3 skipped, 2253 subtests passed`) with
    Ruff `E9,F`, compile, package, shell, and diff gates. `npm run check` passed;
    the reviewed task validator increased total production JavaScript to
    `524487 / 525312` gzip bytes while the entry remained
    `284701 / 448512`. No model, media, network artifact, or generated asset was
    used.
- [x] **P2.2 Existing image paths**
  - Stable Diffusion XL basics; direct Flux img2img/inpaint/ControlNet; Flux
    Kontext multi-reference; Z-Image img2img; supported Qwen img2img,
    edit/inpaint, ControlNet, Edit Plus, and Layered modes; existing registered
    Modular image pipelines.
  - [x] **P2.2a Stable Diffusion XL base text-to-image**
    - Evidence 2026-08-12: the backend now owns an exact
      `StableDiffusionXLPipeline` text-to-image execution specification and a
      content-addressed 1024px planning graph pinned to reviewed revision
      `462165984030d82259a11f4367a4eed129e94a7b`. The client consumes the
      backend-owned revision through the generic `defaultRevision` binding,
      revalidates static node schemas before resealing a refreshed graph, and
      fails closed for malformed revisions or tampered schemas. The template is
      planning-only, Auto-disabled, and Gallery-hidden; no model, media, or
      public asset was downloaded or generated. Pair-scoped generation and
      verification passed for the one SDXL workflow. The complete backend suite
      passed (`1243 tests, 3 skipped`), with the focused graph/catalog matrix at
      `130/130` plus Ruff `E9,F`, compile, package, shell, and diff gates.
      `npm run check` passed with production JavaScript at
      `525295 / 525312` gzip bytes and the entry at `284983 / 448512`.
      Static preflight also reported that this checkout's managed CPU
      environment records an older dependency-contract digest and requires the
      documented repair command before it can be used for future live proof;
      that local environment drift is not source or qualification evidence.
  - [x] **P2.2b Stable Diffusion XL image-to-image**
    - Evidence 2026-08-12: the existing SDXL Studio model now exposes a separate
      `edit_image` execution profile whose exact loader class is
      `StableDiffusionXLImg2ImgPipeline`, while text-to-image retains
      `StableDiffusionXLPipeline`. Both mode receipts bind the same reviewed base
      repository revision without presenting the implementation classes as two
      user-facing models. The canonical edit graph requires one source image,
      binds the generic Edit node at strength `0.65`, and remains planning-only,
      Auto-disabled, and Gallery-hidden. Pair-scoped generation and deterministic
      verification passed for the one new workflow. The complete backend suite
      passed (`1243 tests, 3 skipped`) with Ruff `E9,F`, compile, package, shell,
      and diff gates; `npm run check` passed with production JavaScript at
      `525303 / 525312` gzip bytes and the entry at `284983 / 448512`. No model,
      source media, inference output, or public asset was downloaded or
      generated. Preflight again reported only the already-recorded stale local
      managed-CPU contract digest, so this slice makes no live qualification
      claim.
  - [x] **P2.2 task-contract planning foundation**
    - Evidence 2026-08-12: canonical workflow generation now prefers a curated
      Studio recipe when one exists and otherwise materializes the exact
      backend-owned task-template skeleton. Pending SDXL text-to-image and
      image-to-image graphs were regenerated through their content-addressed
      task contracts, so further P2 modes no longer require placeholder Gallery
      entries, prompts, or per-model client builders. Both pair-scoped
      generators and deterministic verifiers passed; backend graph/catalog and
      task-contract tests passed (`13 passed, 150 subtests`), the complete
      client gate passed before the final redundant-template removal, and the
      final focused template/task/graph suites plus build and bundle gate passed
      at `525120 / 525312` total gzip bytes. No model, media, inference output,
      or public asset was downloaded or generated.
  - [x] **P2.2c Stable Diffusion XL inpaint**
    - Evidence 2026-08-12: the existing logical SDXL model now owns an exact
      `inpaint` execution profile selecting
      `StableDiffusionXLInpaintPipeline` at the same reviewed base revision.
      Its task contract and canonical graph require separate source-image and
      mask inputs, use the generic Inpaint action, and bind the backend-owned
      immutable revision without a static planning template or client
      model-family graph branch. The contract-only standard adapter entry was
      removed once this exact pair became supported; the separate internal
      Modular SDXL path remains contract-only. Pair generation and deterministic
      verification passed. The complete backend suite passed (`1263 passed, 3
      skipped, 2289 subtests`) with Ruff `E9,F`, compile, package, shell, and
      diff gates; `npm run check` passed at `525128 / 525312` total production
      JavaScript gzip bytes with the entry at `284983 / 448512`. No weights,
      source media, inference output, or public asset was downloaded or
      generated, and Auto/Gallery activation remains deferred to P2.5.
  - [x] **P2.2d FLUX.1-dev image-to-image**
    - Evidence 2026-08-12: the existing logical `FluxDevPipeline` model now
      exposes a separate `edit_image` execution profile selecting the exact
      `FluxImg2ImgPipeline` loader at reviewed FLUX.1-dev revision
      `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`. Its task contract and canonical
      graph require one source image, use the generic Edit action, and bind the
      immutable revision without a static planning template or client
      model-family graph builder. Auto remains enabled only for the already
      qualified text-to-image mode; image-to-image remains Expert planning-only
      and Gallery-hidden pending P2.5 qualification. Pair generation and
      deterministic verification passed. The complete backend suite passed
      (`1263 passed, 3 skipped, 2301 subtests`) with Ruff `E9,F`, compile,
      package, shell, and diff gates; `npm run check` passed at
      `525142 / 525312` total production JavaScript gzip bytes with the entry at
      `284983 / 448512`. No weights, source media, inference output, or public
      asset was downloaded or generated.
  - [x] **P2.2e FLUX.1-dev inpaint**
    - Evidence 2026-08-12: the same logical `FluxDevPipeline` model now owns an
      exact `inpaint` execution profile selecting `FluxInpaintPipeline` at the
      reviewed FLUX.1-dev revision. Its task contract and canonical graph
      require separate source-image and mask inputs, use the generic Inpaint
      action, and bind the immutable revision without a static planning
      template or client model-family graph builder. The standard adapter left
      the contract-only registry only after the pair became executable. Auto
      remains text-to-image-only, while direct inpaint stays Expert
      planning-only and Gallery-hidden pending P2.5 qualification. Pair
      generation and deterministic verification passed. The complete backend
      suite passed (`1263 passed, 3 skipped, 2313 subtests`) with Ruff `E9,F`,
      compile, package, shell, and diff gates; `npm run check` passed at
      `525140 / 525312` total production JavaScript gzip bytes with the entry at
      `284983 / 448512`. No weights, source media, inference output, or public
      asset was downloaded or generated.
  - [x] **P2.2f Z-Image Turbo image-to-image**
    - Evidence 2026-08-12: the existing logical `ZImageModularPipeline` Studio
      model now exposes a separate `edit_image` execution profile selecting the
      standard `ZImageImg2ImgPipeline` adapter at reviewed Z-Image Turbo
      revision `f332072aa78be7aecdf3ee76d5c247082da564a6`. Its task contract and
      canonical graph require one source image, use the generic Edit action,
      and bind the immutable revision without a static planning template or
      client model-family graph builder. The standard adapter left the
      contract-only registry only after the pair became executable. Auto
      remains text-to-image-only, while image-to-image stays Expert
      planning-only and Gallery-hidden pending P2.5 qualification. Pair
      generation and deterministic verification passed. The complete backend
      suite passed (`1263 passed, 3 skipped, 2325 subtests`) with Ruff `E9,F`,
      compile, package, shell, and diff gates; `npm run check` passed at
      `525152 / 525312` total production JavaScript gzip bytes with the entry at
      `284983 / 448512`. No weights, source media, inference output, or public
      asset was downloaded or generated.
  - [x] **P2.2g Qwen-Image-2512 image-to-image**
    - Evidence 2026-08-12: the existing logical `QwenImageModularPipeline`
      Studio model now exposes an `edit_image` execution profile selecting the
      standard `QwenImageImg2ImgPipeline` adapter at reviewed Qwen-Image-2512
      revision `25468b98e3276ca6700de15c6628e51b7de54a26`. Its task contract and
      canonical graph require one source image, use the generic Edit action,
      retain the reviewed Qwen Expert resource policy, and bind the immutable
      revision without a static planning template or client model-family graph
      builder. The standard adapter left the contract-only registry only after
      the pair became executable. Existing Auto modes remain unchanged;
      image-to-image stays Expert planning-only and Gallery-hidden pending P2.5
      qualification. Pair generation and deterministic verification passed.
      The complete backend suite passed (`1263 passed, 3 skipped, 2337
      subtests`) with Ruff `E9,F`, compile, package, shell, and diff gates;
      `npm run check` passed at `525154 / 525312` total production JavaScript
      gzip bytes with the entry at `284983 / 448512`. No weights, source media,
      inference output, or public asset was downloaded or generated.
  - [x] **P2.2h Qwen-Image-2512 inpaint**
    - Evidence 2026-08-12: the same logical `QwenImageModularPipeline` Studio
      model now owns an exact `inpaint` profile selecting the standard
      `QwenImageInpaintPipeline` adapter at the reviewed Qwen-Image-2512
      revision `25468b98e3276ca6700de15c6628e51b7de54a26`. Its task contract and
      canonical graph require separate source and mask inputs, use the generic
      Inpaint action, retain the reviewed Qwen Expert resource policy, and bind
      the immutable revision without a static template or client model-family
      graph builder. The adapter left the
      contract-only registry only for this exact mode; its unqualified outpaint
      alias was not advertised. Existing Auto modes remain unchanged, while
      inpaint stays Expert planning-only and Gallery-hidden pending P2.5
      qualification. Pair generation and deterministic verification passed.
      The complete backend suite passed (`1263 passed, 3 skipped, 2349
      subtests`) with Ruff `E9,F`, compile, package, shell, and diff gates;
      `npm run check` passed at `525153 / 525312` total production JavaScript
      gzip bytes with the entry at `284983 / 448512`. No weights, source media,
      inference output, or public asset was downloaded or generated.
  - [x] **P2.2 existing-image-path closure**
    - Evidence 2026-08-13: a manifest-derived audit deterministically verified
      all 30 registered image model/mode pairs. The audit refreshed stale
      canonical layouts for Qwen Image Control and both Qwen Image Edit Plus
      modes. Canonical generation now reapplies reviewed immutable revisions
      from the backend artifact catalog and derives every required Hub artifact
      from the exported graph, preserving the Qwen base, ControlNet Union, and
      Lightning LoRA identities through future regeneration. The focused
      backend graph/catalog, discovery, truth, and task-contract gate passed
      (`32 passed, 230 subtests passed`); the complete client check passed at
      `525153 / 525312` total production JavaScript gzip bytes with the entry at
      `284983 / 448512`. The complete backend source suite had already passed
      for the final P2.2h slice (`1263 passed, 3 skipped, 2349 subtests`). No
      weights, media, live inference, or public assets were used, and remote
      Gallery qualification remains isolated to P2.5.
- [x] **P2.3 Existing audio paths**
  - Stable Audio and existing ACE-Step modes using the generic audio nodes.
  - Evidence 2026-08-13: Stable Audio Open 1.0 moved from the
    contract-only registry into an exact `stable-audio:direct` execution
    profile selecting `StableAudioPipeline` at reviewed revision
    `f21265c1e2710b3bd2386596943f0007f55f802e`. Its planning graph uses only
    the generic Diffusers audio loader/generator/export nodes and binds Stable
    Audio's native task, steps, guidance, waveform-count, duration, and sample
    rate controls. It remains Expert-only and Gallery-hidden pending P2.5 live
    qualification. Pair-scoped generation and verification now include all
    variants for a model/mode pair; all five canonical audio pairs and both ACE
    text-to-audio LoRA variants passed deterministic verification. Regeneration
    refreshed the three ACE text-to-audio layouts and preserved their exact
    base/LoRA artifact receipts. The complete backend suite passed (`1264
    passed, 3 skipped, 2361 subtests passed`) with Ruff `E9,F`, compile,
    package, shell, and diff gates; `npm run check` passed at
    `525276 / 525312` total production JavaScript gzip bytes with the entry at
    `285027 / 448512`. No weights, source media, inference output, or public
    asset was downloaded or generated.
- [x] **P2.4 Existing short-video graph paths**
  - Wan 2.2 I2V/TI2V, Wan Animate, Wan first/last-frame, LTX long-prompt I2V,
    LTX2 joint audio/video, and Hunyuan FramePack.
  - This segment commits graph/template contracts only. It does not run video on
    the local machine.
  - Evidence 2026-08-13: six reviewed planning adapters add ten exact generic
    task contracts: Wan 2.2 A14B text-to-video; both Wan Animate character
    modes; Wan first/last-frame image-to-video; LTX long-prompt
    image-to-video; all four LTX2 video modes with joint video/audio export;
    and Hunyuan FramePack image-to-video. The existing Wan 2.2 I2V and TI2V
    paths remain covered by the same global library audit. All 58 canonical
    pairs and 70 supported workflows, including refreshed deterministic FLUX
    and Z-Image variants, passed graph/hash/layout verification. The backend
    suite passed (`1264 passed, 3 skipped, 2463 subtests`) together with Ruff,
    package, and shell checks. The complete client check passed, the focused
    graph/store suites passed (`80 tests`), and the exact-spec plus pending
    dynamic-schema browser cases passed after the full mocked sweep reported
    `101 passed` and exposed that fixture expectation. The final production
    JavaScript bundle is `525155 / 525312` gzip bytes with the entry at
    `277958 / 448512`. The standalone local preflight remains non-ready only
    because the already-installed CPU profile digest predates the checkout;
    no environment repair, weights, source media, inference output, or public
    asset was required. Auto and Gallery remain disabled pending P2.5.
- [ ] **P2.5 Remote Gallery qualification and activation**
  - Generate examples remotely from the paired commits.
  - Review and publish media to an immutable Dataset revision.
  - Commit descriptors, hashes, rights/provenance, quality reviews, activation,
    and the generated client mirror separately.
  - [x] **P2.5a Clean-host campaign readiness:** client `8a93cf2` makes the
    existing release-qualification campaign usable on a freshly provisioned
    host whose local Auto history and ignored qualification output directory do
    not exist yet. Missing history is treated as absent legacy evidence rather
    than invented proof, and the report creates only its ignored evidence
    directory before writing. A regression exercises that exact clean-host
    boundary. The complete client gate passes with the unchanged 530,915-byte
    production JavaScript gzip total. A campaign dry run now enumerates 76
    missing qualification receipts in six reusable model-family batches
    (ACE-Step, FLUX, LTX, Qwen Image, Wan, and Z-Image). It submitted no graph,
    generated no media, and does not satisfy the remote output, human review,
    Dataset, activation, or physical macOS gates above.
  - [x] **P2.5b Exact app-cache readiness:** client `a76ee04` adds a read-only
    campaign preflight that compares every selected template's backend-derived
    model and LoRA receipt with the running app's bounded `/hf_cache`
    inventory. It accepts only an uncredentialed loopback HTTP(S) origin and
    fails closed for a missing repository, wrong immutable revision,
    non-installed or incomplete entry, repair requirement, malformed response,
    missing artifact receipt, or mismatched execution server. Against the
    current app it reports all 76 pending qualification jobs ready across all
    31 unique exact artifacts. The focused nine-test campaign matrix and the
    complete client gate pass; production JavaScript remains 530,915 gzip
    bytes. This is cache-readiness evidence only: no graph was submitted, no
    model was executed, and no output, review, publication, activation, or
    physical macOS evidence is claimed.
  - [x] **P2.5c Exact default-input readiness:** client `d271a9f`, corrected by
    `7738537`, extends the campaign preflight to the byte-pinned Template
    Gallery defaults used by each selected job. It validates the
    content-addressed runtime path against
    the binding digest and checked asset manifest, rejects absent, linked,
    oversized, size-mismatched, or hash-mismatched local files, and stops the
    campaign before browser or inference startup when any input is unavailable.
    The focused 12-test matrix and complete client gate pass; production
    JavaScript remains 530,915 gzip bytes. The corrective slice checks both the
    lightweight authoring location and the normal installer's durable backend
    `web/` payload, and the runner resolves the same installed-app fallback.
    The current source checkout reports
    38 input-free jobs ready and 38 jobs blocked by 50 absent exact inputs
    totaling 33,867,388 bytes, while all 31 model/LoRA artifacts remain
    app-ready. The absent payload was not downloaded outside the app and no
    existing model was removed. A normal installer-managed Gallery payload is
    still required on the approved qualification app host before P2.5 remote
    execution begins; no graph, inference, media, review, publication,
    activation, or physical macOS evidence is claimed.
  - [x] **P2.5d App-owned pinned Gallery materialization:** backend `df71942`
    and client `0fd0830` add the normal running-app path needed to close the
    absent-input condition without bypassing the app. Setup now exposes strict
    status, plan, and explicit install/repair actions for the exact anonymous
    Dataset revision. The backend validates the immutable source descriptor
    and canonical manifest identity, reserves the complete download and
    same-volume staging copy alongside active model-download reservations and
    the 64 GiB safety margin, hashes every staged byte, and atomically promotes
    the verified tree. Admission is serialized with model-download space
    reservations, while transfers retain the app's bounded parallelism. This
    action never deletes model-cache entries. The reviewed approved-subset
    descriptor resolves to 356 assets / 480,430,370 bytes and
    `sha256:canonical-json:5ec869b755a6ce04a789d6835819da150493bfaef8a6bc1480f0274ba05bcab9`.
    Focused backend tests passed (43 tests / 29 subtests), the complete backend
    gate passed (`1625 passed, 40 skipped, 3273 subtests`), the complete client
    gate and bundle budget passed, and the full mocked Studio sweep passed all
    107 tests. The currently running app predates these routes and was not
    restarted because app-managed model downloads remain active; no Gallery
    install POST or payload download has occurred. Until those downloads
    finish, the new app code is activated by a safe restart, and the user
    explicitly confirms the in-app plan, the present source-checkout result
    remains 38 input-free jobs ready and 38 input-conditioned jobs blocked. No
    graph, inference, media, review, publication, activation, or physical macOS
    evidence is claimed.
  - [x] **P2.5e Bounded parallel app downloads:** backend `c313908` closes a
    mismatch between the server's two-slot download semaphore and the Hub
    transfer boundary. The prior process-wide Xet lock wrapped every complete
    snapshot call, so independently admitted ordinary downloads still ran one
    at a time. A writer-preferring shared/exclusive mode gate now lets two
    ordinary app-owned snapshots use the existing bounded slots concurrently,
    while a repair waits for all normal transfers to drain, blocks new ones,
    temporarily disables process-global Xet behavior, and restores its exact
    prior value before ordinary work resumes. Queue-aware immutable byte
    reservations and the 64 GiB safety margin are unchanged, and this path
    deletes no cache entries. The focused download matrix passes 44 tests;
    repeated overlap tests prove two normal transfers enter together, a third
    stays queued, and repair mode never leaks in either admission order. The
    complete backend gate passes (`1627 passed, 40 skipped, 3273 subtests`),
    Ruff E9/F, 66-package compatibility, shell/diff checks, and portable
    preflight pass; preflight intentionally observed the healthy existing app
    on port 8088 rather than claiming a free-port startup. That running worker
    predates this commit and was not restarted while its already-admitted
    downloads remain active, so this is source/unit evidence for the next safe
    worker restart, not a claim that the current transfers changed mode or that
    any model/media/macOS qualification completed.
  - [x] **P2.5f Bounded app-owned Hub transport:** backend `77ed298` moves
    model and Gallery snapshot payloads onto the standard Hub HTTP path, whose
    per-request timeouts and retries provide a bounded failure boundary while
    preserving completed cache blobs. Model and Gallery work now share the
    app's existing two-transfer semaphore. Ordinary payload transfers may
    overlap within that limit; repair is writer-exclusive from its first cache
    preparation step and restores the exact prior process-global Hub transport
    policy after the exclusive window drains. Queue-aware exact byte plans,
    immutable revisions, the 64 GiB reserve, and the no-deletion policy remain
    unchanged. The focused matrix passes 56 tests, the concurrency/repair/
    Gallery race subset passes five repeated runs, and the complete backend
    gate passes (`1630 passed, 40 skipped, 3273 subtests`) together with Ruff
    E9/F, 66-package compatibility, shell/diff checks, and portable preflight.
    The already-running worker still predates this source change and was not
    restarted; its existing AuraFlow transfer and app-owned overnight queue
    were left untouched. This records source/unit behavior only: no active
    transfer was switched, no model or Gallery payload was deleted, and no
    generation, review, Dataset publication, activation, remote, or physical
    macOS evidence is claimed.

### Phase 2 test and asset gate

- [x] Backend graph/catalog/profile integrity tests pass.
- [x] Client template, quality, Gallery coverage, and mocked browser tests pass.
- [ ] Every public template has a remote live-output receipt for its exact mode.
- [ ] Every media byte is in the Dataset, not either Git repository.
- [ ] Auto remains disabled for any template whose qualification is pending.

## Phase 3 — Small, fast pipelines and speech recognition

Priority: first new live execution. Hardware: allowlisted local smoke or remote.
Assets: generated remotely even when a local smoke is allowed.

### Committable segments

- [x] **P3.1 Generic unconditional image generation**
  - Add an unconditional task mode/adapter, not DDPM-specific nodes.
  - Integrate `DDPMPipeline`, `DDIMPipeline`, and
    `ConsistencyModelPipeline` in separate exact-pair entries.
  - Local smoke: tiny resolution and bounded steps; hard timeout 40 minutes.
  - Evidence 2026-08-13: backend `80e4587` and client `8f2a671`
    add one generic `UnconditionalGenerate` adapter, three immutable exact
    model/mode pairs, prompt-free fixed-resolution Studio controls, and three
    deterministic canonical workflows. `google/ddpm-cifar10-32` is pinned at
    `267b167dc01f0e4e61923ea244e8b988f84deb80` for DDPM and DDIM;
    `openai/diffusers-cd_imagenet64_l2` is pinned at
    `5f462e4403fc37b72ec6004e806c71805db22387` for the consistency
    model. Offline cached CPU node smokes completed at two DDPM steps
    (`32x32`, `0.10s`), two DDIM steps (`32x32`, `0.08s`), and one
    consistency step (`64x64`, `0.28s`), all far below the 40-minute bound.
    The backend gate passed (`1266 passed, 3 skipped, 2511 subtests`) with
    Ruff `E9,F`, package, shell, and diff checks; all 73 canonical workflows
    verified; `npm run check` and the complete mocked Studio browser sweep
    passed (`102 passed`). The production JavaScript bundle is
    `525941 / 526336` total gzip bytes with the entry at
    `278350 / 448512`. Auto and Gallery remain disabled pending remote output
    review and immutable Dataset publication. No generated media was retained
    or committed. The standalone preflight confirmed port 8088 was free but
    remains non-ready only because the installed CPU profile digest predates
    this checkout, the same unrelated local drift recorded for P2.4.
- [ ] **P3.2 Small latent image workflows**
  - Stable Diffusion 1.x/2.x text-to-image, img2img, and inpaint.
  - LCM 1-4 step workflows and PAG using compatible base weights.
  - Local smoke: at most 512px and the minimum meaningful step count.
  - [x] **P3.2a Stable Diffusion 1.5 exact generic workflows**
    - Evidence 2026-08-13: backend `a0815b8` and client `5a633a9`
      add immutable text-to-image, img2img, and inpaint pairs through the
      existing generic image nodes. All three pairs reuse
      `stable-diffusion-v1-5/stable-diffusion-v1-5` at
      `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` with safetensors and
      CreativeML Open RAIL-M provenance; no pipeline-specific execution node
      or client graph builder was added.
    - Offline cached CPU node smokes completed at `64x64`: text-to-image at
      one step in `0.68s`, img2img at two requested steps (`strength=0.8`, one
      effective denoise step) in `0.81s`, and inpaint at two steps in `0.92s`.
      The backend gate passed (`1266 passed, 3 skipped, 2562 subtests`) with
      Ruff `E9,F`, package, shell, and diff checks; all 76 canonical workflows
      verified; `npm run check` and the complete mocked Studio browser sweep
      passed (`103 passed`). The production JavaScript bundle is
      `526012 / 526336` total gzip bytes with the entry at
      `278350 / 448512`.
    - Auto and Gallery remain disabled pending remote quality review and
      immutable Dataset publication. No generated media was retained or
      committed. The standalone preflight confirmed port 8088 was free but
      remains non-ready only because the installed CPU profile digest predates
      this checkout, the same unrelated local drift recorded for P2.4 and
      P3.1.
  - [ ] **P3.2b Stable Diffusion 2.x exact generic workflows**
    - Access review 2026-08-13: the official Stability AI 2.x repositories
      remain gated to this unauthenticated qualification environment. No
      unreviewed replacement repository or invented immutable revision was
      admitted. This slice remains pending independently and does not block
      the qualified 1.5, LCM, PAG, or perception work.
    - App-plan recheck 2026-08-14: both known official immutable candidates,
      `stabilityai/stable-diffusion-2-1-base@1f758383196d38df1dfe523ddb1030f2bfab7741`
      and
      `stabilityai/stable-diffusion-2-inpainting@218a32afeef278c9ed76affdcb6ea16b653cc8c0`,
      still return repository-not-found/gated API responses with unknown size.
      No download was submitted and no mirror was substituted.
  - [x] **P3.2c Latent Consistency Model 1-4 step workflows**
    - Evidence 2026-08-13: backend `6a1b579` and client `9f7b7da`
      add one exact generic text-to-image pair for
      `SimianLuo/LCM_Dreamshaper_v7` at immutable revision
      `a85df6a8bd976cdd08b4fd8f3b73f229c9e54df5`, using the official
      `LatentConsistencyModelPipeline`, safetensors, and MIT provenance. The
      Studio profile exposes its native one-to-four-step range with a
      four-step quality default and no unsupported negative-prompt control.
    - The exact component snapshot cached in `6m58s`; the offline cached CPU
      `LoadPipeline` plus generic `Generate` node smoke loaded in `0.57s` and
      completed one step at `64x64` in `0.49s`. The in-memory output digest was
      `fed2db181ee91a8edb6823f2959d37fc841ebd07196281232fcc9cf56112f508`;
      no output file was retained.
    - The backend gate passed (`1266 passed, 3 skipped, 2578 subtests`) with
      Ruff `E9,F`, package, shell, and diff checks; all 77 canonical workflows
      verified; `npm run check` and the complete mocked Studio browser sweep
      passed (`103 passed`). The production JavaScript bundle is
      `526082 / 526336` total gzip bytes with the entry at
      `278350 / 448512`. Auto and Gallery remain disabled pending remote
      quality review and immutable Dataset publication. The standalone
      preflight confirmed port 8088 was free and remains non-ready only for the
      previously recorded local CPU profile digest drift.
  - [x] **P3.2d PAG workflows using compatible base weights**
    - Evidence 2026-08-13: backend `662aa10` and client `42c4dd6` add one
      exact generic text-to-image pair for the official
      `StableDiffusionPAGPipeline`. It reuses the reviewed
      `stable-diffusion-v1-5/stable-diffusion-v1-5` safetensors artifact at
      immutable revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` and
      exposes bounded `pag_scale` and `pag_adaptive_scale` aliases through the
      existing generic image generator. Selecting the PAG profile initializes
      the upstream-compatible `3.0` and `0.0` defaults and writes both values
      into the exact generated graph.
    - The offline cached CPU `LoadPipeline` plus generic `Generate` node smoke
      loaded in `0.57s` and completed one step at `64x64` in `0.54s` with
      guidance `7.5`, PAG scale `3.0`, and adaptive scale `0.0`. The in-memory
      output digest was
      `69644cbb6d528adf9e18625bf17d3ce126ff54014b3af7380329787ebf45cfd6`;
      no output file was retained.
    - The backend gate passed (`1266 passed, 3 skipped, 2596 subtests`) with
      Ruff `E9,F`, package, shell, compile, and diff checks; all 78 canonical
      workflows verified. `npm run check` passed and the complete mocked
      Studio browser sweep passed (`104 passed`), including the exact PAG
      control/default/binding flow. The production JavaScript bundle is
      `526453 / 527360` total gzip bytes with the entry at
      `278617 / 448512`. Auto and Gallery remain disabled pending remote
      quality review and immutable Dataset publication. The standalone
      preflight confirmed port 8088 and required imports are ready but remains
      non-ready only for the previously recorded local CPU profile digest
      drift (`74d79558...` installed versus `60aa03fa...` current).
- [x] **P3.3 Generic perception output**
  - Add prediction-map output semantics and integrate Marigold depth first.
  - Add normals, intrinsics, and uncertainty only after the shared output
    contract is stable.
  - Evidence 2026-08-13: backend `957ab31` and client `1802291` add the
    generic `Predict Map` boundary and exact Marigold depth Studio workflow.
    The reviewed Apache-2.0 repository
    `prs-eth/marigold-depth-lcm-v1-0` is pinned to immutable revision
    `04a73502f7fd8fc5e59947b9df3b2266d71d6849`; the node returns a
    schema-versioned normalized float32 NHWC relative-depth map and a
    grayscale preview. Processing resolution and input-resolution matching
    are explicit bounded graph bindings. Normals, intrinsics, and uncertainty
    remain deliberately unavailable rather than being inferred from depth.
  - The exact safetensors/config snapshot was installed with legacy unsafe
    weights excluded. An offline cached CPU `LoadPipeline` plus generic
    `PredictMap` smoke loaded in `0.330s` and completed one step at `64x64` in
    `0.254s`. The in-memory prediction had shape `[1, 64, 64, 1]`, range
    `[0.0, 1.0]`, and SHA-256
    `3cd1069551742ea3ec3c79e1897207c1f5bc97b6bd1bffad3b316b3866ce4911`;
    no output file was retained.
  - The backend gate passed (`1269 passed, 3 skipped, 2618 subtests`) with
    Ruff `E9,F`, package, shell, compile, and diff checks; all 79 canonical
    workflows verified. `npm run check` passed and the complete mocked Studio
    browser sweep passed (`105 passed`). Auto and Gallery remain disabled
    pending remote quality review and immutable Dataset publication. The
    standalone preflight remains non-ready only for the previously recorded
    local CPU profile digest drift.
- [x] **P3.4 Adopt the official Hugging Face library boundary in repository policy**
  - Update backend and client `AGENTS.md`, contributor guidance, Hugging Face
    standards, dependency/runtime contracts, and the former Diffusers-only
    boundary tests.
  - Permit reviewed official Hugging Face libraries while preserving the single
    MoDiff graph executor, local execution, immutable sources, no implicit
    remote code, generic contracts, rollback, and proof requirements.
  - This is a documentation/contract commit and generates no media.
  - Status 2026-08-14: backend `91c9a36` and client `28b12b7` implement and
    validate the paired policy and executable-boundary contract. The direct
    Transformers and PEFT dependency migration gap recorded at that checkpoint
    was subsequently closed by the platform-scoped P0.5 cutover.
- [x] **P3.5 Hugging Face Transformers speech-to-text implementation**
  - Add generic `Load Speech Recognition Model` and `Transcribe Audio` nodes.
  - Support transcription, optional translation, language hint, timestamps, and
    chunking through a normalized contract.
  - Begin with immutable safetensors revisions of Whisper Tiny/Base or another
    reviewed Hugging Face ASR model. Do not create Whisper-specific nodes.
  - Local smoke: a short rights-approved fixture; hard timeout 40 minutes.
  - Security tests: path/media validation, duration/size limits, no remote code,
    bounded output, cleanup, and offline cached execution.
  - Package the Transformers runtime through P0.5; do not restore it to default
    application dependencies.
  - Evidence 2026-08-13: backend `82522ba` and client `571facf` add generic
    `Load Speech Recognition Model` and `Transcribe Audio` nodes plus exact
    prompt-free transcription and English-translation Studio workflows. The
    reviewed Apache-2.0 `openai/whisper-tiny` safetensors snapshot is pinned to
    immutable revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`.
    The normalized schema-v1 result contains the task, text, timestamp mode,
    duration, and bounded segments; language hints, none/segment/word
    timestamps, and bounded chunk/stride controls are explicit graph bindings.
  - The loader remains lazy, sets `trust_remote_code=False`, requires
    safetensors, and uses the existing P0.5 Transformers+PEFT optional-runtime
    profile rather than adding a base dependency. Focused tests cover managed
    path validation, file/duration/sample-rate/channel limits, non-finite
    media, immutable revisions, offline cached loading, malformed results, and
    aggregate transcript limits. The current base observation remains
    `wrong_version` because it contains Transformers 5.15.0; the exact
    Transformers 5.14.1 boundary was instead installed into an isolated
    temporary target for the live proof and removed afterward.
  - An offline cached CPU smoke passed through the real MoDiff loader and
    action using a two-second rights-safe signal authored in memory. Model load
    took `0.532s`, transcription took `0.144s`, and the normalized no-timestamp
    result contained 20 text characters with SHA-256
    `d54010ab982673ed0c6ddc41683626f456463ced73d1d768658e70f1487188f7`;
    no fixture or output media was retained. Semantic/quality review against a
    rights-reviewed spoken fixture remains a remote Dataset gate.
  - The backend gate passed (`1274 passed, 3 skipped, 2646 subtests`) with Ruff
    `E9,F`, package, shell, compile, and diff checks; all 81 canonical workflows
    verified. `npm run check` passed and the complete mocked Studio browser
    sweep passed (`106 passed`). The production JavaScript bundle is
    `528630 / 529408` total gzip bytes with the entry at
    `280178 / 448512`. Auto and Gallery remain disabled pending remote speech
    quality review and immutable Dataset publication.

### Phase 3 test and asset gate

- [x] Static signature and artifact-policy tests pass for every admitted exact model/mode.
- [x] Tiny/mocked output normalization tests pass.
- [x] Paired client forms, graph bridges, readiness, errors, and browser flows
  pass.
- [x] Each local smoke completes below 40 minutes or is moved to remote without
  a local retry.
- [ ] Remote media and ASR fixtures pass rights review and Dataset verification.
- [x] Only exact live-qualified recipes may enter Auto.

## Phase 4 — Medium image, audio, and 3D integrations

Priority: after the small-model contracts are stable. Hardware: remote by
default. Assets: remote Dataset only.

### Committable segments

- [ ] **P4.1 Control adapters:** SD1.5 ControlNet and T2I Adapter with pinned
  preprocessors and auxiliary models.
  - [x] **P4.1a SD1.5 ControlNet Canny:** the generic Diffusers image loader
    assembles `StableDiffusionControlNetPipeline` from the existing immutable
    SD1.5 base plus `lllyasviel/control_v11p_sd15_canny` commit
    `115a470d547982438f70198e353a921996e2e819`. Both repository loads require
    safetensors, the auxiliary kind/class/parameter and independent revision
    are backend-owned, and the admission receipt binds the complete assembly.
    The canonical graph runs the existing generic Canny node at exact
    thresholds `0.1/0.2` before `ControlGenerate`; a CPU tiny fixture verifies
    the preprocessor extent and non-empty output. The generated workflow is the
    82nd deterministic catalog entry. Auto, Gallery, and live proof remain off
    pending remote output review.
  - [ ] **P4.1b SD1.5 T2I Adapter:** deferred independently. The reviewed
    official `TencentARC/t2iadapter_canny_sd15v2` snapshot at
    `a18baf4f0ff002f34dc2f19c4b93fe00d4cce9ee` publishes legacy PyTorch `.bin`
    weights rather than safetensors. The generic loader rejects
    `StableDiffusionAdapterPipeline`; no unsafe-deserialization exception or
    unlicensed community conversion was admitted.
  - P4.1a source commits are backend `539650a` and client `785b43e`. The final
    backend gate passed (`1280 passed, 3 skipped, 2661 subtests`) with Ruff
    `E9,F`, package, shell, compile, and diff checks. All 82 workflows verify;
    `npm run check` passed, the mocked Studio sweep passed (`106 passed`), and
    shared-control browser coverage passed (`2 passed`). The production bundle
    remains within budget at `528981 / 529408` total gzip bytes and
    `280348 / 448512` for the entry chunk. No weights or output media were
    downloaded or retained.
- [x] **P4.2 SDXL expansion:** Turbo first, then the reviewed text, image,
  inpaint, instruct, ControlNet, adapter, PAG, and related combinations.
  - [x] **P4.2a SDXL Turbo text-to-image:** the generic Diffusers image loader
    exposes a logical `StableDiffusionXLTurboPipeline` backed by the upstream
    `StableDiffusionXLPipeline` and immutable `stabilityai/sdxl-turbo` commit
    `71153311d3dbb46851df1931d3ca6e939de83304`. Loading is restricted to the
    reviewed fp16 safetensors variant, inference is bounded to one through four
    steps, guidance is fixed to zero, and negative prompting is hidden. The
    canonical 512px one-step graph is the 83rd deterministic catalog entry.
    Auto, Gallery, live output, license-surface approval, and physical macOS
    qualification remain pending; no result is inferred from Linux static or
    mocked evidence.
  - [x] **P4.2b SDXL InstructPix2Pix image editing:** the generic Diffusers
    image loader exposes the upstream
    `StableDiffusionXLInstructPix2PixPipeline` against immutable
    `diffusers/sdxl-instructpix2pix-768` commit
    `06653d47f8d22f2c2205a5884d6a24c5e76d2ca7`. Loading requires the reviewed
    safetensors-only snapshot without remote code. The exact experimental
    recipe accepts one source image at 768px, 30 steps, text guidance 3, and
    image guidance 1.5; the new backend-owned generic image-guidance field is
    bounded to the upstream-supported range and forwarded without a
    model-specific client branch. Its canonical graph is the 84th
    deterministic catalog entry. Auto, Gallery, live output, and physical
    macOS qualification remain pending; no output quality claim is inferred
    from contract or mocked evidence.
  - [x] **P4.2c SDXL ControlNet Canny:** the generic conditioned Diffusers image
    loader assembles the immutable SDXL base with
    `diffusers/controlnet-canny-sdxl-1.0` commit
    `eb115a19a10d14909256db740ed109532ab1483c`. Both loads require the reviewed
    fp16 safetensors variants; the auxiliary artifact receipt additionally
    records the exact 2,502,139,136-byte file and SHA-256. The upstream
    `StableDiffusionXLControlNetPipeline` receives the existing generic Canny
    preprocessor output at exact thresholds 0.1/0.2 and the reviewed 1024px,
    50-step, guidance-5, conditioning-scale-0.5 recipe. Its canonical graph is
    the 85th deterministic catalog entry. Auto, Gallery, live output, and
    physical macOS qualification remain pending; no output quality claim is
    inferred from contract or mocked evidence.
  - [x] **P4.2d SDXL T2I-Adapter Canny:** the same generic conditioned image
    contract now assembles the immutable SDXL base with Apache-2.0
    `TencentARC/t2i-adapter-canny-sdxl-1.0` commit
    `2d7244ba45ded9129cfbf8e96a4befb7f6094210`. The reviewed fp16 component is
    the exact 158,060,440-byte `diffusion_pytorch_model.fp16.safetensors` file
    with SHA-256
    `e3db0d9cb3dd54c116a429a1de067d952780047944dd7dba8be032dd2e737f81`;
    no remote code or unsafe deserialization is allowed. The upstream
    `StableDiffusionXLAdapterPipeline` receives the generic Canny preprocessor
    output at exact thresholds 0.1/0.2 and the reviewed 1024px, 30-step,
    guidance-7.5, adapter-scale-0.8 recipe. Its canonical graph is the 86th
    deterministic catalog entry. Auto, Gallery, live output, and physical
    macOS qualification remain pending; no output quality claim is inferred
    from contract or mocked evidence.
  - [x] **P4.2e SDXL PAG text-to-image:** the generic Diffusers image loader
    exposes upstream `StableDiffusionXLPAGPipeline` over the existing immutable
    `stabilityai/stable-diffusion-xl-base-1.0` commit
    `462165984030d82259a11f4367a4eed129e94a7b`. Loading remains restricted to
    the reviewed fp16 safetensors variant with no remote code and no auxiliary
    artifact. The backend-owned recipe follows the upstream 1024px, 50-step,
    guidance-5 defaults with PAG scale 3 and adaptive scale 0; both PAG controls
    use the existing generic action fields. Its canonical graph is the 87th
    deterministic catalog entry. Auto, Gallery, live output, and physical
    macOS qualification remain pending; no output quality claim is inferred
    from contract or mocked evidence.
  - [x] **P4.2f SDXL PAG image-to-image and inpaint:** the same immutable SDXL
    base now loads the upstream `StableDiffusionXLPAGImg2ImgPipeline` and
    `StableDiffusionXLPAGInpaintPipeline` classes through exact generic image
    adapters. Both modes retain fp16 safetensors-only loading, no remote code,
    and no auxiliary artifact. The reviewed MoDiff recipe uses 1024px, 50
    steps, guidance 5, strength 0.8, PAG scale 3, and adaptive scale 0; edit
    requires one source image and inpaint requires one source plus one mask.
    Their canonical graph hashes are
    `2a74886a72cdf5c66b90a58fc4b31797d2e94073e44c61833da350979a21065f`
    and `6903adcae1149856449ba860706b887214f7493b365cb2264afc0ce4ebc1f762`,
    bringing the deterministic catalog to 89 workflows. Auto, Gallery, live
    output, and physical macOS qualification remain pending; no output quality
    claim is inferred from static, unit, or mocked evidence.
  - Existing SDXL base text-to-image, image-to-image, and inpaint source slices
    remain recorded under P2.2a through P2.2c. Other related combinations
    require independent admission and do not reopen the completed reviewed
    P4.2 set.
  - P4.2a source commits are backend `fb49ed8` and client `f893514`. The final
    backend gate passed (`1282 passed, 3 skipped, 2680 subtests`) with Ruff
    `E9,F`, package, shell, compile, and diff checks. All 83 workflows verify;
    `npm run check` passed and the complete mocked Studio sweep passed
    (`106 passed`). The production bundle remains within budget at
    `529013 / 529408` total gzip bytes and `280348 / 448512` for the entry
    chunk. No weights or output media were downloaded or retained.
  - P4.2b source commits are backend `eb2a28e` and client `b7ed626`. The
    complete backend gate passed (`1283 passed, 3 skipped, 2696 subtests`) with
    Ruff `E9,F`, package, shell, compile, and diff checks. All 84 workflows
    verify; `npm run check` passed and the complete mocked Studio sweep passed
    (`106 passed`). The production bundle remains within budget at
    `529098 / 529408` total gzip bytes and `280348 / 448512` for the entry
    chunk. No weights or output media were downloaded or retained.
  - P4.2c source commits are backend `a4ae9ca` and client `5440570`. The
    complete backend gate passed (`1284 passed, 3 skipped, 2712 subtests`) with
    Ruff `E9,F`, package, shell, compile, and diff checks. All 85 workflows
    verify; `npm run check` passed, shared-control browser coverage passed
    (`2 passed`), and the complete mocked Studio sweep passed (`106 passed`).
    The production bundle remains within budget at `529223 / 529408` total
    gzip bytes and `280348 / 448512` for the entry chunk. No weights or output
    media were downloaded or retained.
  - P4.2d source commits are backend `2d14051` and client `b13d65d`; client
    commit `c267b35` separately fixes the shared portalled-select closed-state
    pointer contract exposed by the browser gate. The complete backend gate
    passed (`1285 passed, 3 skipped, 2728 subtests`) with Ruff `E9,F`, package,
    shell, compile, build, and diff checks. All 86 workflows verify;
    `npm run check` passed, shared-control browser coverage passed (`2 passed`),
    and the previously blocked media-export browser case passed after the
    shared fix. The complete mocked sweep then passed 105/106; its sole
    unrelated supervisor-poll overlap assertion passed immediately in an
    isolated rerun. The production bundle remains within budget at
    `529223 / 529408` total gzip bytes and `280348 / 448512` for the entry
    chunk. The local managed CPU runtime reports a pre-existing source-digest
    drift while package compatibility remains healthy; no runtime was mutated
    for this source slice. No weights or output media were downloaded or
    retained.
  - P4.2e source commits are backend `2a9c29b` and client `379936e`. The
    complete backend gate passed (`1286 passed, 3 skipped, 2745 subtests`) with
    Ruff `E9,F`, package, shell, compile, build, and diff checks. All 87
    workflows verify and `npm run check` passed. The production bundle remains
    within its exact budget at `529378 / 529408` total gzip bytes and
    `280364 / 448512` for the entry chunk. The local managed CPU runtime still
    reports the same pre-existing source-digest drift while package
    compatibility remains healthy; no runtime was mutated for this source
    slice. No weights or output media were downloaded or retained.
  - P4.2f source commits are backend `63f9075` and client `c0f2e2b`. The
    complete backend gate passed (`1286 passed, 3 skipped, 2784 subtests`) with
    Ruff `E9,F`, compile, package, build, and diff checks. All 89 workflows
    verify with deterministic canonical layouts. `npm run check` passed; after
    the final prose-only bundle reduction, the focused 82-test profile suite,
    formatting, production build, and budget check also passed. The production
    bundle remains within budget at `529387 / 529408` total gzip bytes and
    `280364 / 448512` for the entry chunk. The local managed CPU runtime still
    reports only the pre-existing source-digest drift while imports, package
    compatibility, and port availability remain healthy; no runtime was
    mutated. No weights or output media were downloaded or retained.
- [x] **P4.3 Moderate image families:** DreamLite, Sana/Sana Sprint, and other
  candidates admitted by the per-model checklist.
  - [x] **Sana 0.6B text-to-image:** the generic Diffusers image loader exposes
    upstream `SanaPipeline` against immutable
    `Efficient-Large-Model/Sana_600M_1024px_diffusers` commit
    `28f3af7689de15f3883d5863059a2fca0aa9b829`. The reviewed approximately
    7.70 GB snapshot is safetensors-only, requires no repository Python, and
    uses its fp16 variant with the text encoder and VAE placed in bfloat16 as
    documented upstream. The backend-owned recipe is 1024px, 20 steps,
    guidance 4.5, and maximum sequence length 300. Apache-2.0 applies alongside
    the bundled Gemma terms and prohibited-use policy. Its canonical graph hash
    is `3184feed07ae57e6f0cdcc3cca8e07f36d3b205e564c6c8fdea2a83ebc94dc6d`.
  - [x] **Sana Sprint 0.6B generation and editing:** the loader exposes exact
    upstream `SanaSprintPipeline` and `SanaSprintImg2ImgPipeline` classes against
    immutable `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers` commit
    `aa76e7f4f4928f378716b6716a2130fba3caf5b1`. The reviewed approximately
    7.70 GB snapshot is native bfloat16 safetensors, requires no repository
    Python, and is bounded to one through four steps. The canonical recipe uses
    1024px, two steps, guidance 4.5, maximum sequence length 300, and edit
    strength 0.5. Its text and edit graph hashes are
    `1867366ab8b89ff35cb9f09dbe6725b6184af5674df0b6076b5f24a7dc600fbe`
    and `9a9ef604ec5da99da3616dd38b118162e75657c322b80e2c57f53b745b1d9bda`.
    The same Apache/Gemma rights surface remains visible for later review.
  - [x] **DreamLite base and mobile generation/editing:** the reviewed Phase 6
    Diffusers pin exports `DreamLitePipeline` and `DreamLiteMobilePipeline`.
    Exact immutable `diffusers`-branch snapshots
    `carlofkl/DreamLite-base@751cb8dbb9072a8c8ffd8684e0f254b50f20531b`
    and
    `carlofkl/DreamLite-mobile@6695c3f4be230f0493fa5dbf78be3bc4d3bb2ab4`
    are ungated, contain no repository Python, and expose three safetensors
    weight files / 5,040,118,270 bytes each. Both remain non-commercial under
    CC-BY-NC-4.0. Base uses 1024px, 28 steps, text guidance 3.5, edit image
    guidance 1.5, and at most 200 prompt tokens. Mobile is bounded to one
    through eight steps with four recommended; its ignored text/image guidance
    inputs are omitted from the graph rather than presented as functional
    controls. The base text/edit graph hashes are
    `c55d65890ac84a17c61788ab37085055c9479882f7af44c98aaf5382b2c65243`
    and `7b0a0a8f61125ee39cbf40383044ce8959dac5b5950622b46889555bb98ccca5`;
    the mobile text/edit hashes are
    `1f5a2b3b02a4a266133d695a79539191dc7652aa80607e7fb06a839e66da6281`
    and `3ab28cdc0cd0782e3b7a03717c9127d0365a1639bd922b0247fdd4bc0fbec9cc`.
    Upstream does not expose a per-step callback for these pipelines, so exact
    step-level progress/cancellation remains unavailable pending an upstream
    contract; task-level cancellation remains unchanged.
  - P4.3 source commits are backend `7117c80` and `a56e9c0`, client `c41d1c6`
    and `96f444b`; client generator-race fix `2c99769` made canonical graph
    generation wait for authoritative capability discovery. The earlier Sana
    complete backend gate passed (`1288 passed, 3 skipped, 2837 subtests`) with
    Ruff `E9,F`, compile, package, build, and diff checks. DreamLite's clean
    optional-runtime gate passed 158 focused backend tests; compile, JSON,
    workflow, and diff checks passed, and all 103 workflows verify
    deterministically. `npm run check` passed after the final DreamLite and
    generator changes, including 90 template/profile cases and the production
    bundle budget (`529948 / 530432` total gzip bytes and `279775 / 448512` for
    the entry chunk). Auto, Gallery, live output, remote quality, and physical
    macOS qualification remain pending. No weights or output media were
    downloaded or retained.
- [x] **P4.4 Audio generation:** LongCat AudioDiT, Stable Audio quality recipes,
  and AudioLDM2 general audio.
  - [x] **Stable Audio quality contract:** the existing generic
    `StableAudioPipeline` route remains pinned to automatic-gated
    `stabilityai/stable-audio-open-1.0` commit
    `f21265c1e2710b3bd2386596943f0007f55f802e` under the Stability AI
    Community License. Its reviewed bounded recipe remains 30 seconds, 100
    steps, guidance 7, one waveform, and 48 kHz. This slice makes safe
    serialization explicit in the loader. Its canonical graph hash is
    `895eb04d3f30f5468df8c7eee0296963b5980f7f3578e5add0ea03e1239a7256`.
  - [x] **LongCat AudioDiT:** exact upstream
    `LongCatAudioDiTPipeline` support uses the reviewed Diffusers-format
    `ruixiangma/LongCat-AudioDiT-1B-Diffusers` conversion at immutable commit
    `f4c063ea37f262ba5e6129ebd80095a6d6a9de4d`. The approximately 5.70 GB
    MIT-aligned snapshot is ungated, safetensors-only, and contains no
    repository Python. The backend-owned recipe is 5 seconds, 16 steps,
    guidance 4, one waveform, and native 24 kHz, with a hard 30-second bound.
    Its canonical graph hash is
    `294e3be641d069938edb8fa5c631f5a8aef321297795dfb0638e8cf1c8cf5be3`.
  - [x] **AudioLDM2 general audio:** exact upstream `AudioLDM2Pipeline`
    support uses official `cvssp/audioldm2` commit
    `c8e7e189d324425c05c4c2f81214041ef4107983`. The selected approximately
    4.48 GB execution envelope is safetensors-only even though the repository
    also retains legacy `.bin` files; `use_safetensors=True` prevents unsafe
    fallback. CC-BY-NC-SA-4.0 remains visible for later release review. The
    bounded recipe is 10 seconds, 200 steps, guidance 3.5, three waveforms,
    and native 16 kHz. Its canonical graph hash is
    `be80299c1142f548e7504a6276aff313d821b0e44651b1d5337359030363f283`.
  - The generic audio adapter now owns family-specific duration, step,
    guidance, waveform, sample-rate, callback, and output-shape contracts.
    Declarative loader/generator metadata actions remain available without
    activating the optional runtime, while actual model loading and execution
    still require the exact qualified runtime. Client finalization invokes
    those actions and waits for the backend-owned pipeline/task contract before
    sealing a managed graph, fixing the previously stale ACE-default schema for
    non-ACE audio workflows.
  - P4.4 source commits are backend `4d6a4d3` and client `f0958e6`; client
    commit `2c55d7c` separately consolidates the media fallback required to
    retain the strict bundle ceiling. The complete backend gate passed (`1290
    passed, 3 skipped, 2871 subtests`) with Ruff `E9,F`, compile, package,
    build, shell, workflow-generation, and diff checks. All 94 workflows verify
    deterministically. `npm run check` passed, including 84 profile/template
    and 43 graph-visual contract cases; the production bundle is within budget
    at `528923 / 529408` total gzip bytes and `279752 / 448512` for the entry
    chunk. The local managed CPU runtime still reports only the previously
    recorded source-digest drift (`74d795...` installed versus `60aa03...`
    current) while required imports, package compatibility, device validation,
    and port availability remain healthy; no runtime was mutated. Auto,
    Gallery, live output, remote quality, and physical macOS qualification
    remain pending. No weights or output media were downloaded or retained.
- [ ] **P4.5 Diffusers text-to-speech:** deferred independently. The pinned
  Diffusers `AudioLDM2Pipeline` exposes the required generic `prompt` plus
  `transcription` speech-synthesis call contract, and upstream documentation
  identifies `anhnct/audioldm2_gigaspeech` as the GigaSpeech TTS checkpoint.
  The reviewed ungated snapshot at immutable commit
  `c812a7861f38a69441a8e0428438e782d9864614` is approximately 5.83 GB and
  contains legacy `.bin` weights for the language model, projection model,
  both text encoders, UNet, VAE, and vocoder with no safetensors alternative.
  The related `anhnct/audioldm2_ljspeech` snapshot at
  `32ab10ffc92907e6a6741319675a1175fd925a66` has the same unsafe-only
  serialization envelope. No credible immutable safetensors conversion was
  found, and no unsafe-deserialization exception is approved, so MoDiff does
  not expose or execute either TTS repository. Revisit this segment only when
  a reviewed safetensors artifact exists or the owner approves a narrowly
  documented exception with isolated conversion and provenance review.
- [x] **P4.6 Generic 3D artifacts:** Shap-E rendered output is admitted from
  official `openai/shap-e` commit
  `7bd337afdea1c17842e1c3cc45c4e268356dba40` through explicit safe component
  assembly: the fp16 safetensors `prior`, safe text encoder, and the reviewed
  pre-rename safetensors `renderer`. The renamed `shap_e_renderer` directory is
  deliberately excluded because its reviewed snapshot exposes only legacy
  `.bin` weights. The backend seals a bounded 1-120 frame, 64-256px rendered
  orbit and rejects unsafe artifact identity, quantization, or device-map
  overrides; mesh/PLY/OBJ/GLB remain unavailable until a separate safe export
  contract exists. The canonical graph hash is
  `da3650ce1a40f12e29760ee1ea58baaab3edb4f57b89683339be76ed4400d7c0`.
  Source commits are backend `21d819f` and client `d684fc4`. The complete
  backend gate passed (`1297 passed, 3 skipped, 2888 subtests`) with Ruff
  `E9,F`, compile, package, build, shell, workflow-generation, and diff checks.
  All 95 workflows verify deterministically; `npm run check` passed with 86
  profile/template and 43 graph-visual contract cases, and the production
  bundle remains within budget at `529351 / 529408` total gzip bytes and
  `279914 / 448512` for the entry chunk. The 106-case mocked Studio sweep has
  complete passing evidence: 101 cases passed in the full run and the five
  stale audio-mock failures all passed after the mock was aligned with the
  backend-owned action contract. Auto, Gallery, live output, remote quality,
  and physical macOS qualification remain pending. No model weights or output
  media were downloaded or retained.

### Phase 4 test and asset gate

- [x] Unit and tiny-fixture tests cover adapters, outputs, and cleanup for each
  completed Phase 4 segment.
- [x] Backend/client integrated gates pass for each completed independent segment.
- [x] No Phase 4 live model is required to run locally.
- [ ] Remote receipts include peak memory, runtime, dependency/model revisions,
  graph hash, media checks, and cleanup result.
- [ ] Gallery activation follows rights and anonymous byte verification.

## Phase 5 — Short video qualification

Priority: after image/audio contracts. Hardware and assets: remote only.

### Committable segments

- [ ] Qualify the existing Wan, LTX/LTX2, and Hunyuan FramePack graph paths from
  Phase 2 using minimal short outputs.
- [x] Add Stable Video Diffusion using documented offload and decode chunking.
  The source-qualified image-to-video slice pins the official gated
  `stabilityai/stable-video-diffusion-img2vid-xt-1-1` repository at commit
  `043843887ccd51926e3efed36270444a838e7861`, admits only its reviewed
  safetensors artifact surface, and exposes the Stability AI Community License
  gate before download. The adapter follows the documented CPU-offload,
  UNet-forward-chunking, and two-frame decode-chunk recipe; rejects prompt,
  source-video, mask, unsafe identity, quantization, and device-map overrides;
  and seals bounded image conditioning through exact execution contract
  `stable-video-diffusion:image-to-video:v1`. The canonical graph hash is
  `2bfcddd0ae5a6214c891f1859180abfc04a3ce0d51f47d73d7aa503910df8ef0`.
  Source commits are backend `260637d` and client `d6e0eed`. The complete
  backend gate passed (`1302 passed, 3 skipped, 2911 subtests`) with Ruff
  `E9,F`, package, shell, JSON, workflow-generation, and diff checks. All 96
  workflows verify deterministically; `npm run check` passed with 43
  graph-visual cases, five consecutive exact-contract regression runs, and all
  106 mocked Studio cases. The production bundle remains within budget at
  `529077 / 529408` total gzip bytes and `279720 / 448512` for the entry chunk.
  No model weights or output media were downloaded or retained. Remote output,
  quality, Dataset, Auto, Gallery, and physical macOS qualification remain
  pending.
- [x] Add AnimateDiff/AnimateLCM with separately pinned base model,
  `MotionAdapter`, scheduler rules, and optional LoRA. The Expert-only source
  slice reuses the reviewed safetensors SD1.5 base at
  `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`; pins AnimateDiff v1.5.2 at
  `6167b88ffe39b4441fdf2113e77b99a6f56b7906` and AnimateLCM at
  `3d4d00fc113225e1040f4d3bec504b6ec750c10c`; and loads only the exact fp16
  `MotionAdapter` safetensors plus the named
  `AnimateLCM_sd15_t2v_lora.safetensors` file. The loader seals the documented
  linear-beta DDIM/LCM scheduler recipes, AnimateLCM adapter scale `0.8`, VAE
  slicing, model CPU offload, fixed 512px output, 8-16 frames, and bounded
  steps/guidance. It rejects mismatched base/motion revisions, unsafe artifact
  substitution, media conditioning, quantization, device maps, and malformed
  outputs before execution can be admitted. The two canonical graph identities
  are `a39d33676afba9bcfd364c5f9a1d8c3afb146a537e82d8818babfe156528c4cf`
  and `658dff27b265437ef3528b6cdb46d7a7bd865b69727329658603bda7804ae726`;
  their checked-in file SHA-256 values are
  `4b54e1ddb53c4f40f3c44d8bb272c0c5703aef2c3ecbd01fd3e55a29298bd39e`
  and `f30938d1fc02c836f057cdeb838199d9d13161c12afc8c280edeb5b02e04d426`.
  Source commits are backend `75dde3c` and client `220fb40`. The complete
  backend gate passed (`1306 passed, 3 skipped, 2937 subtests`) with Ruff
  `E9,F`, package, shell, JSON, and diff checks. All 98 workflows verify
  deterministically; `npm run check` and all 106 mocked Studio cases passed.
  The production bundle remains within its reviewed ceiling at
  `529711 / 530432` total gzip bytes and `279775 / 448512` for the entry chunk.
  Neither motion repository declares a weight license, so the client records an
  explicit rights-undetermined acknowledgment and MoDiff grants no use rights.
  Auto and Gallery remain disabled; no weights or output media were downloaded
  or retained. Authorization review, remote execution/quality, Dataset, Auto,
  Gallery, and physical macOS qualification remain pending.
- [x] Evaluate Motif Video independently before source admission. The official
  public Apache-2.0 `Motif-Technologies/Motif-Video-2B` snapshot was reviewed at
  immutable commit `6748a1f5861a859aca13b30a0c65dd6f9942ca35`. Although the
  denoiser is described as 2B, its safetensors weight surface is approximately
  17.26 GB: an 8,599,946,488-byte text encoder, 8,151,344,960-byte transformer,
  and 507,591,892-byte VAE. The official native recipe is 121 frames at
  1280x736 and 50 steps, and the documented constrained-memory path requires
  model CPU offload. That total artifact and native decode envelope is not a
  smaller local candidate, so executable admission is deferred to a remote
  heavy-model review with measured peak accelerator/system memory. No weights
  or media were downloaded and no executable or client surface was added.
- [x] Evaluate CogVideoX-2B independently after artifact-size and RAM review.
  The Expert-only source slice pins the official public Apache-2.0
  `zai-org/CogVideoX-2b` repository at immutable commit
  `1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01`. Its reviewed safetensors-only
  weight surface is 13,774,687,212 bytes across the two T5 shards, transformer,
  and VAE, with every file size and SHA-256 recorded in the artifact catalog.
  The admitted graph is deliberately shorter than the publisher's
  representative 49-frame/50-step recipe: exact native 720x480 output, 9-25
  frames in `4k+1` form, 1-50 steps, guidance 1-12, and a maximum prompt length
  of 226. Loading requires float16, the exact artifact, safetensors, mandatory
  VAE tiling, and model CPU offload; it rejects media conditioning,
  quantization, device maps, alternate artifacts, multiple outputs, and
  non-PIL output. The canonical graph identity is
  `6b9b9badc24068955b8beab40f3f8990292bd70d0c9dfd46dc5d295133245316`
  and its checked-in file SHA-256 is
  `ad2669a7c80d69dcda610d81cb894373c869ca199764edf3a877b83f690b39fc`.
  Source commits are backend `6501e22` and client `00a3802`. The focused gate
  passed (`139 passed, 823 subtests`) against the clean Linux base; the complete
  backend gate passed in an isolated reviewed Transformers/PEFT test overlay
  (`1311 passed, 3 skipped, 2957 subtests`) while the delivered base remained
  free of both optional packages and preflight-ready. Ruff `E9,F`, package,
  compile, JSON, workflow, and diff checks passed. All 99 workflows verify
  deterministically; `npm run check` and all 106 mocked Studio cases passed.
  The production bundle remains within its reviewed ceiling at
  `529827 / 530432` total gzip bytes and `279775 / 448512` for the entry chunk.
  No weights or output media were downloaded or retained. Remote runtime,
  memory, output/quality, Auto, Gallery, Dataset, and physical macOS proof
  remain pending.

### Phase 5 test and asset gate

- [x] Static and mocked tests cover frame count, dimensions, conditioning,
  scheduler/adapter compatibility, output normalization, and cleanup for each
  completed Phase 5 source slice.
- [ ] Remote smoke uses the minimum supported 8-25 frames and bounded steps.
- [ ] Representative quality proof is limited to approximately 2-4 seconds.
- [ ] Non-black frames, finite tensors, duration/frame rate, and decode/mux
  integrity are checked without cross-hardware pixel hashes.
- [ ] Auto remains disabled until the exact short-video recipe has live proof.

## Phase 6 — Pin update, heavy models, and long-form workflows

Priority: last. Hardware and assets: dedicated remote qualification only.

### Committable segments

- [x] Review all commits between the current and proposed Diffusers pins; update
  the executable dependency, compatibility test, and upstream contract tests in
  one isolated change. The exact 73-commit delta and green compatibility
  evidence are recorded above; source commit is backend `5ee9e1d`, with the
  unchanged client contract revalidated.
- [x] Add the post-pin Krea2 and Krea2 Turbo Modular classes. Backend
  `1753384` and client `ec2a349` register both as Expert-visible,
  contract-only image surfaces. The pinned upstream contracts remain distinct:
  base uses `Krea2AutoBlocks`, 28 steps, negative prompting, and a guider;
  Turbo uses `Krea2TurboAutoBlocks`, 8 steps, and neither negative prompting
  nor a guider. Both expose only required-prompt `text_to_image`, have no
  runnable mode, artifact, template, Auto, or Gallery surface, and are rejected
  before artifact resolution. The deterministic snapshot now contains 28
  contracts and 80 upstream workflows; 1,313 backend tests plus 2,965 subtests,
  the complete client check, and all 106 mocked Studio cases pass. No weights
  were downloaded.
- [x] Complete the standard Krea 2 Raw/Turbo artifact, package, license, and
  admission-gate review without accepting repository terms. Backend `e176f14`
  seals exact official revisions
  `krea/Krea-2-Raw@6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603` and
  `krea/Krea-2-Turbo@98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7`.
  Each 55-file repository has a Python-free 17-file Diffusers candidate
  partition with five safetensors files / 35,666,644,396 weight bytes; exact
  weight identities, metadata blob receipts, duplicate root-native checkpoint
  exclusions, pinned standard and Modular package source hashes, distinct
  28-step/guidance-4.5 Raw and 8-step/guidance-free Turbo recipes, and
  estimate-only resource envelopes are sealed in
  `data/krea2-artifact-review.json` without fetching model-weight bytes.

  Admission remains blocked rather than silently accepting rights on the
  user's behalf. Both Hugging Face repositories require affirmative acceptance
  of the Krea 2 Community License and its moving Acceptable Use Policy. The
  pinned license limits commercial use to entities below USD 1 million in
  trailing company-wide annual revenue unless an enterprise license is
  obtained, carries recipient-acceptance/model-naming/license-copy/notice
  distribution duties, and mandates reasonable deployment content filtering.
  The package has no safety checker and no step, sequence-length, or output
  pixel ceiling. The app-only plans were inspected but not submitted: with
  528,238,714,880 free bytes, a 450,696,547,273-byte existing queue, and the
  68,719,476,736-byte reserve, Raw and Turbo were respectively short by
  53,182,524,488 and 53,172,350,108 bytes. No older model was deleted.

  The complete backend overlay passes 1,631 tests, 3,580 subtests, and three
  platform skips; the focused gate passes 21 tests and 104 subtests, Ruff E9/F
  and package compatibility pass, and no client change is required. Legal and
  product approval, task-scoped terms acceptance, downstream terms/filter
  implementation, immutable AUP evidence, bounded runtime admission, app
  capacity, remote real-weight review, and physical macOS execution remain
  independent gates.
- [x] Complete the standard Stable Diffusion 3 artifact, package, license, and
  admission-gate review without accepting repository terms. Backend `e6061d9`
  seals exact official revision
  `stabilityai/stable-diffusion-3-medium-diffusers@ea42f8cef0f178587cf766dc8129abd379c90671`.
  Its Python-free 38-file snapshot occupies 31,012,147,557 bytes. The exact
  six-file fp16 partition contains 15,499,002,486 weight bytes and reuses the
  immutable base inventory already sealed by the SD3 ControlNet review; no
  base weight or safetensors-header bytes were fetched for this slice. Exact
  artifact identities, pinned text-to-image, image-to-image, and inpaint
  pipeline source hashes, native 1024px recipes, callbacks, interrupt checks,
  CPU-offload contract, and estimate-only resource bounds are sealed in
  `data/stable-diffusion-3-artifact-review.json`.

  Admission remains fail-closed. The repository requires affirmative
  acceptance of the Stability AI Non-Commercial Research Community License,
  and production, hosted-service, and API use require a separate license. The
  authenticated model index and component configs remain HTTP 401 without
  acceptance; MoDiff did not accept those terms. The package pipelines have no
  safety checker and do not bound steps, input pixels, or output pixels. The
  app-only plan was inspected but not submitted: 524,908,945,408 free bytes
  minus the 446,582,359,749-byte existing queue and 68,719,476,736-byte reserve
  left 9,607,108,923 bytes, making the 31,012,147,557-byte snapshot short by
  21,405,038,634 bytes. No direct weight download occurred and no older model
  was deleted.

  The focused boundary matrix passes 165 tests and 516 subtests. The complete
  backend overlay passes 1,636 tests, 3,580 subtests, and three platform skips;
  Ruff E9/F and package compatibility pass, and no client change is required.
  Task-scoped terms acceptance, commercial-license and legal/product approval,
  authenticated component review, backend-owned limits and runtime admission,
  app capacity, remote real-weight output review, and physical macOS execution
  remain independent gates.
- [x] Add `MiniMaxH3ModularPipeline` only through generic joint video+audio
  specifications for its distinct `t2va`, `fl2va`, and `ref2va` workflows.
  Validate the `transformer/` versus `transformer_ref/` partition receipt,
  Qwen3-VL conditioning, separate video/audio scheduler state, reference-media
  bounds, immutable artifact revision, and remote-only resource envelope before
  exposing any mode. Backend `baf7271` and client `947a2ef` complete the
  contract-only slice without exposing a runnable mode or default repository.
  The three upstream workflows map to generic `text_to_video_with_audio`,
  `first_last_frame_to_video_with_audio`, and
  `reference_to_video_with_audio`; the generic contract schema now preserves
  `fl2va`'s `prompt + image` or `prompt + last_image` alternatives instead of
  collapsing them into an invalid conjunction. The immutable official snapshot
  is `MiniMaxAI/MiniMax-H3` at
  `42ed227ee7df40d41602854ae760620d6eb651fe`. Its root Modular surface is 46
  safetensors files and 210,296,909,532 weight bytes with both transformer
  partitions; one workflow selects 77,735,901,100 shared bytes plus exactly one
  66,280,504,216-byte transformer, for 144,016,405,316 weight bytes. Every file
  size and SHA-256 is sealed in `data/minimax-h3-artifact-review.json`, while
  duplicated legacy `FL2VA/`, `Ref2VA/`, and publisher media are excluded.
  The receipt also seals Qwen3-VL layer 50, video/audio scheduler shifts 12/3,
  24 fps and `17*n+5` frame alignment, 5-15-second output bounds, and the
  9-image/3-video/3-audio/12-total reference limits with audio-only reference
  requests forbidden. Its MiniMax H3 Community License excludes the US, EU,
  UK, and Republic of Korea and adds commercial/redistribution obligations, so
  the repository is deliberately absent from MoDiff's runtime/download catalog
  pending legal and remote hardware qualification. The remote resource envelope
  is estimate-only, with at least 160 GiB selective-workflow disk, 256 GiB
  system RAM, 192 GiB aggregate accelerator memory, and four accelerators; no
  live claim is made. The complete backend suite passes at 1,318 tests plus
  2,969 subtests, the complete client check and bundle gate pass, and all 106
  mocked Studio cases pass. No weights or media were downloaded.
- [x] Add the post-pin `LTX2ModularPipeline` and `LTX25ModularPipeline` and
  complete their no-weight source/recipe review. Backend `7e4f99b` and client
  `9dfcf62` register both as Expert-visible, contract-only multimodal surfaces,
  with four generic joint video/audio workflows each: text, image, condition,
  and in-context generation. Both publish the shared conditioner, duration
  head, audio VAE, and vocoder contracts and return video plus audio; in-context
  generation requires an explicit frame count. LTX-2 retains convolutional VAE
  decode controls, while LTX-2.5 replaces them with the diffusion decoder.
  Neither class has a runnable mode, default repository, Auto path, template,
  or Gallery surface, and both fail closed before artifact resolution. The
  deterministic snapshot now contains all 31 reviewed Modular contracts and 91
  upstream workflows. The complete client check and focused mocked-browser
  contract test pass; no weights or media were downloaded.

  Backend `b887aef` separately seals the immutable LTX-2.5 source review in
  `data/ltx-2.5-artifact-review.json`. It binds distilled single-stage to the
  exact eight-sigma schedule with guidance disabled; full/SFT stage 1 to dynamic
  shifting plus x2 latent upsampling, the distilled LoRA, and the exact
  three-sigma stage-2 tail while retaining stage-1 audio; and distilled
  two-stage to the same eight/three sigma schedules with one generator and audio
  latents carried into stage 2. The receipt also binds the duration head,
  48 kHz audio-VAE/vocoder handoff, diffusion decode with tiled NATTEN attention
  and seeded `denormalize=False`, and the separate Gemma-4 enhancement recipe.
  Prompt enhancement and NATTEN kernel provisioning are explicit execution/setup
  actions and may not download during discovery or planning.
- [ ] Complete LTX-2.5 artifact and live recipe qualification after gated access
  is granted and the LTX-2.x Community License is accepted. The official
  `Lightricks/LTX-2.5-Diffusers` snapshot is pinned at
  `a6de4b5354f078db24d9cf4778c14846788aea3d`; its public metadata seals 31
  safetensors files and 163,896,920,128 bytes, but the repository denies file
  access on this machine. Consequently the gated component indexes cannot yet
  resolve the concurrent four-shard/eight-shard `transformer/` layouts, no exact
  selective partition or remote resource envelope is claimed, and the artifact
  remains outside both runtime and download catalogs. The reviewed license
  requires a paid commercial agreement at USD 10 million aggregate annual
  entity revenue and places obligations on derivative transfers. Remote heavy
  execution and physical macOS evidence remain pending independently.
- [x] Evaluate HunyuanVideo 1.5 without widening its existing contract-only
  surface. Backend `31cafc4` seals the official
  `tencent/HunyuanVideo-1.5` snapshot at
  `9b49404b3f5df2a8f0b31df27a0c7ab872e7b038` and two immutable Diffusers
  candidates: 480p text-to-video at
  `286be7ce72277246578a3e3cc2487e95ddae5bcf` and 480p step-distilled
  image-to-video at `854c04a4c8a53d990b418c7478f0802c0fc8c726`.
  Their exact safetensors receipts are respectively 371,759,988,572 bytes for
  the full upstream family, 53,367,753,676 bytes for the selective T2V
  repository, and 34,620,593,582 bytes for the selective I2V repository. The
  source recipes bind 121 frames at 24 fps, 50 steps / guidance 6 / scheduler
  shift 5 for T2V, and the recommended 8-or-12-step mean-flow path with
  guidance 1 / shift 7 for I2V. CPU model offload, VAE tiling, and optional
  attention kernels are recorded; prompt rewriting and kernel downloads remain
  forbidden during discovery.

  No runtime or download entry was admitted. The Tencent Hunyuan Community
  License preamble excludes the EU, UK, and South Korea, while the formal
  `Territory` definition names only the EU; it also adds a 100-million-MAU
  commercial threshold, distribution notice, generated-content disclosure,
  and output-territory obligations. That inconsistency requires legal review.
  Remote execution and physical macOS evidence remain pending, and the resource
  envelope in `data/hunyuanvideo-1.5-artifact-review.json` is estimate-only.
- [x] Evaluate Helios/Pyramid without widening the existing contract-only
  surface. Backend `873f0ce` pins Helios Base, Mid, and Distilled at
  `5c50b6bc90eae9bd815d2a50b0c9877e3fd2cf88`,
  `477c55427ec0ea774bdebd0fbe736313cfc5a312`, and
  `b991c0379a018f4de3227d95468237f56066f5bb`. Each repository contains 18
  safetensors files / 137,730,908,420 bytes, but its declared standard index
  selects only the shared text encoder and VAE plus `transformer/`: 12 files /
  80,481,086,028 bytes. The extra 57,249,822,392-byte `transformer_init/` or
  `transformer_ode/` partition is recorded and excluded from the candidate
  runtime surface.

  The three existing Modular contracts retain exact generic text-to-video,
  image-to-video, and video-to-video workflows at the reviewed 384x640 and
  132-frame defaults. The source receipt separately seals Base's 99-frame,
  50-step, guidance-5 recipe and Distilled's 240-request/264-rounded frame,
  `[2, 2, 2]` pyramid-step, guidance-1, amplify-first-chunk recipe at 24 fps.
  Publisher claims for approximately 6 GB group-offload memory and 19.5 H100
  fps are explicitly not live qualification evidence. No runtime or download
  entry was admitted: every pinned upstream `modular_model_index.json` embeds
  `revision: null` for all downloadable components, and the roughly 80.48 GB
  selected partitions still require remote heavy-hardware execution. Physical
  macOS evidence remains pending independently; no weights or media were
  downloaded.
- [x] Evaluate Wan 2.2 A14B Modular without changing the separately registered
  standard adapters. Backend `011a70b` seals the existing official T2V A14B
  snapshot at `5be7df9619b54f4e2667b2755bc6a756675b5cd7` and I2V A14B
  snapshot at `596658fd9ca6b7b71d5057529bbf319ecbc61d74`. Their exact
  safetensors surfaces are 28 files / 126,177,598,620 bytes and 28 files /
  126,180,875,420 bytes respectively: one shared T5 encoder, one Wan VAE, and
  separate twelve-shard high- and low-noise 14B experts. No weights were
  downloaded.

  Neither repository publishes `modular_model_index.json`. At the pinned
  Diffusers revision the immutable standard indexes instead map
  `WanPipeline` plus `boundary_ratio=0.875` to `Wan22ModularPipeline`, and
  `WanImageToVideoPipeline` plus `boundary_ratio=0.9` to
  `Wan22Image2VideoModularPipeline`. The receipt binds that fallback without
  remote code, the exact contract-only no-required-input T2V and required-image
  I2V workflows, and the distinct source recipes: 720x1280 / 81 frames / 40
  steps / guidance 4+3 for T2V, and input-aspect 480x832-area / 81 frames / 40
  steps / inherited guidance 3.5 for I2V, both exported at 16 fps. The
  publisher's 80 GB native GPU statement is not treated as live evidence.
  Modular runtime admission remains pending remote fallback-assembly and
  heavy-hardware qualification; physical macOS evidence remains independently
  pending. The existing standard graph-only adapters and artifact pins are
  unchanged.
- [x] Evaluate the full classic LTX/LTX2 artifact surfaces and remove the
  identity-changing classic fallback. Backend `0f96a92` removes
  `Lightricks/LTX-Video` from all four 13B Distilled execution profiles: at the
  pinned `8984fa25007f376c1a299016d0957a37a2f797bb` revision that family
  repository's standard index selects a two-shard 2B transformer, so it cannot
  silently replace the six-shard 13B default. It remains pinned in the artifact
  catalog for explicit review but is no longer a published execution candidate.

  Backend `474b83d` seals three immutable inventories. The 13B Distilled
  repository at `7c64400e1861cc0d7b98d570a1926d5408ec60cd` contains 21
  safetensors files / 92,762,951,244 bytes, while its standard index selects 11
  files / 47,628,403,428 bytes and excludes nested duplicate encoder/transformer
  shards. The classic family repository contains 27 files /
  253,809,013,320 bytes but selects only 7 files / 28,419,691,124 bytes. LTX-2
  at `47da56e2ad66ce4125a9922b4a8826bf407f9d0a` contains 44 files /
  314,290,794,056 bytes; its exact pipeline partition is 23 files /
  92,034,380,210 bytes after its model indexes exclude eight root alternatives,
  the concurrent twelve-shard Diffusers-named Gemma layout, and the separately
  invoked latent upsampler. Canonical full and selected inventory digests,
  component index hashes, and exact sizes are recorded without downloading
  weights.

  The existing two-workflow `LTXModularPipeline` contract and all eight classic
  LTX/LTX2 graph surfaces remain unchanged. The LTX-2 two-stage source recipe is
  bound at 768x512, 121 frames, 24 fps, 40-step guidance-4 latent generation,
  then x2 latent upsampling and a three-step guidance-1 distilled tail that
  reuses stage-1 audio latents. The classic 13B Distilled card's Diffusers
  example targets the dev repository and its distilled YAML link points to the
  dev YAML, so it is explicitly not accepted as exact recipe proof. Classic
  immutable license text, LTX-2 commercial-license acceptance, remote
  heavy-hardware execution, and physical macOS evidence remain pending. The
  full backend gate passes (`1,341 passed, 3 skipped, 2,981 subtests`) with Ruff
  `E9,F` and a dependency-clean optional overlay.
- [x] Evaluate EasyAnimate V5.1 without publishing an unqualified runtime or
  download surface. Backend `3c6da73` seals three public, ungated,
  Apache-2.0 Diffusers conversions at immutable revisions: 7B text-to-video at
  `f605a9340b46e725da4c1953dc891884dd694314`, 12B inpaint at
  `8a257d883449752ecaa6bd4990caf932e927de33`, and 12B control at
  `4daad26e8f7f701b37148ac91dc36a4981303af2`. Their exact safetensors
  partitions are respectively 7 files / 31,189,417,564 bytes, 7 files /
  41,159,117,300 bytes, and 7 files / 41,159,485,940 bytes. All share the same
  five-shard Qwen2-VL text encoder and Magvit VAE, while their distinct
  transformers use 16, 33, and 48 input channels; no fallback substitution is
  allowed between them.

  The receipt binds the pinned Diffusers source and documentation contract:
  256-1024 dimensions, 1-49 frames with 49 preferred, and 8 fps export. It
  separately records the exact source examples for 512x512 T2V with 50 steps
  and guidance 6, 448x576 image conditioning through
  `get_image_to_video_latent`, and 672x384 control through
  `get_video_to_video_latent`. The inpaint and control tensor-preparation paths
  still require reviewed MoDiff media adapters, and all three 31-41 GB
  candidates require remote heavy-hardware execution. Therefore the family is
  intentionally absent from runtime and download catalogs; resource envelopes
  are estimate-only, physical macOS evidence remains pending independently,
  and no weights or media were downloaded.
- [x] Evaluate the full SkyReels V2 Diffusers surface without admitting an
  ambiguous-license or unqualified long-form runtime. Backend `5b53632` seals
  all eight official immutable conversions: two 14B T2V, three I2V, and three
  diffusion-forcing repositories across 540p and 720p families. Their exact
  safetensors partitions range from 8 files / 28,973,450,372 bytes for 1.3B
  diffusion forcing through 21 files / 91,339,283,940 bytes for 14B I2V.
  Canonical inventory digests, component totals, model indexes, transformer
  configurations, and the distinct 16-channel T2V/DF versus 36-channel I2V
  contracts are recorded without downloading weights.

  The pinned source exposes base T2V/I2V plus diffusion-forcing T2V/I2V/V2V.
  The reviewed DF recipe binds 97-frame windows, 30 denoising steps per
  five-latent-frame block, `ar_step=5`, five blocks, and 50 total scheduler rows;
  the source-only long-form example advances three windows over 257 frames with
  17-frame overlaps. That is not live qualification. The pinned documentation's
  first/last-frame and V2V examples instead target the unlisted and publicly
  unresolvable `SkyReels-V2-DF-1.3B-720P-Diffusers`, while the separate V2V
  source example omits its required `video` argument.

  No runtime or download entry was admitted. Each model snapshot embeds only a
  short notice linking a mutable PDF; the exact linked PDF was frozen at its
  source commit for review, but it defines the licensed model as
  `Skywork-13B`, not SkyReels V2. Legal scope review, a corrected immutable
  long-form recipe, remote heavy-hardware execution, and physical macOS
  evidence remain pending. Resource envelopes are estimate-only; no media was
  generated or committed.
- [x] Evaluate Cosmos3 Omni and Distilled without widening their existing
  contract-only surfaces. Backend `8ba89fa` seals the public Nano snapshot at
  `411f42a8fdfb8c5b2583cb8786e0938f49796eaa`, Super at
  `e0262be9d8f7586bc24c069a2aed2b665bdff266`, Super T2I 4-Step at
  `0573a4b26b8e15d13d416e51f4680c8bc8b8c33d`, and Super I2V 4-Step
  at `81da615b7f92dc710c6359b072beee06f675979c`. Their exact
  safetensors surfaces are respectively 10 files / 34,894,818,144 bytes, 30 /
  132,624,780,208, 29 / 131,391,926,304, and 28 / 129,405,417,712.
  Component totals, canonical inventory digests, index hashes, and the
  36-layer Nano versus 64-layer Super transformer contracts are recorded
  without downloading weights.

  The existing Modular discovery contracts remain unchanged: Omni exposes ten
  text/image/video, optional-sound, and action workflows; Distilled exposes
  four vision workflows with the exact four-sigma schedule, guidance fixed at
  1, and negative prompts ignored. The full source recipe is sealed at
  720x1280, 189 frames, 24 fps, 35 steps, guidance 6, UniPC flow shift 10,
  while prompt upsampling remains an explicit external action forbidden during
  discovery. Task-pipeline safety defaults on; Modular use additionally calls
  `enable_safety_checker()` and depends on `cosmos_guardrail`.

  No runtime or download entry was admitted. Nano/Super Modular descriptors
  set every component revision to null, distilled descriptors omit revision,
  and the Nano/Super standard indexes name `Cosmos3OmniDiffusersPipeline`,
  which the pinned Diffusers build does not export. Both distilled standard
  indexes configure no safety checker. OpenMDW 1.1 is linked but not embedded;
  its reviewed mutable web response is recorded with distribution-notice and
  litigation-termination terms. Immutable descriptors, class resolution,
  guardrail and remote heavy-hardware qualification, and physical macOS
  evidence remain pending; all resource envelopes are estimate-only.
- [x] Evaluate gated Cosmos 1, Predict2, Predict2.5, and Transfer2.5 as
  metadata-only candidates. Backend `7c6c29c` seals eight immutable
  safetensors manifests: Cosmos 1 7B Text2World and Video2World; Predict2 2B
  Text2Image plus 2B/14B Video2World; Predict2.5 2B post-trained; Transfer2.5
  2B general; and its edge ControlNet. Their exact surfaces range from one
  942,523,208-byte ControlNet file through 12 files / 41,564,530,104 bytes for
  Cosmos 1 Video2World. Component totals, canonical manifest digests, branch
  revisions, inaccessible model-index Git blob identities, and the pinned
  Diffusers source contracts are recorded without downloading weights.

  The source recipes remain evidence only: Cosmos 1 uses 704x1280 / 121 frames
  / 36 steps / 30 fps; Predict2 uses 704x1280 / 93 frames / 35 steps / 16 fps;
  Predict2.5 uses 93 frames / 36 steps / 16 fps; Transfer2.5 edge control uses
  93-frame chunks, 36 steps, guidance 3, and control scale 1. All repositories
  require click-through license acceptance before file bodies resolve. The
  receipt independently hashes the January, April, and September 2025 NVIDIA
  Open Model License prompts and preserves their license/notice, `Built on
  NVIDIA Cosmos`, Trustworthy AI, and safety-guardrail obligations. No runtime
  or download surface was added. License acceptance, gated component-index and
  linked Trustworthy AI review, remote heavy execution, and physical macOS
  evidence remain pending; resource envelopes are estimate-only.
- [x] Evaluate Kandinsky 5 Video without admitting a remote-heavy runtime.
  Backend `08e2550` seals all ten public official Diffusers snapshots at
  immutable revisions: Pro T2V/I2V plus Lite SFT, no-CFG, distilled-16-step,
  and pretrain variants for both 5-second and 10-second generation. Every
  selected artifact is safetensors-only and MIT-declared. The exact selected
  surfaces are 8 files / 23,854,029,536 bytes for each Lite snapshot,
  8 files / 62,666,834,400 bytes for Pro T2V, and 8 files /
  96,526,404,752 bytes for Pro I2V; component hashes, canonical manifest
  digests, model indexes, transformer configurations, and the shared
  19,280,899,008-byte text-encoder/CLIP/VAE partition are recorded without
  downloading weights.

  Pinned source contracts and recipes are sealed for 512x768 Lite generation,
  121-frame 5-second and 241-frame 10-second runs at 24 fps, guidance 1 for
  no-CFG/distilled variants, and the 16-step distilled schedule. Pro T2V's
  documented 768x1024 recipe and required FlexAttention/compile/offload setup
  are evidence only. The pinned video guide's nominal image-to-video example
  incorrectly constructs the T2V pipeline and never supplies its loaded image;
  the pipeline source instead requires `Kandinsky5I2VPipeline` and an `image`
  argument. No runtime or download entry was admitted. Corrected upstream
  recipe evidence, remote heavy-hardware execution, and physical macOS proof
  remain pending; resource envelopes are estimate-only and no media was
  generated.
- [x] Evaluate Kandinsky 2.1 and keep its composite runtime fail-closed.
  Backend `c04f151` seals the public decoder, shared prior, and inpaint decoder
  at three distinct immutable revisions. Their selected safetensors-only
  surfaces are respectively 3 files / 7,428,873,150 bytes, 3 files /
  5,798,697,734 bytes, and 3 files / 7,428,942,270 bytes; duplicate legacy
  `.bin` weights are excluded. Canonical inventory digests, bounded
  safetensors headers, immutable metadata, all seven package pipeline sources,
  and the upstream Apache-2.0 license receipt are recorded without downloading
  full weights.

  No runtime, download, capability, graph, or client surface was admitted. The
  combined package loader copies the decoder's immutable `revision` to the
  connected prior repository, where that distinct commit does not exist; its
  download path instead follows the prior's moving default branch. The prior
  executes before the decoder-only legacy callback and has neither a callback
  nor interrupt flag, so cancellation does not cover the full job. Missing
  backend composite revision binding and bounds, absent output safety checks,
  model-snapshot license-file clarification, remote execution, live output
  review, and physical macOS evidence remain independent gates.
- [x] Evaluate Kandinsky 2.2 without weakening composite or serialization
  policy. Backend `75bb0ac` seals all five official versioned repositories:
  decoder, prior, inpaint decoder, depth ControlNet, and decoder refiner. The
  first three expose exact safetensors surfaces of 2 files / 5,283,689,948
  bytes, 3 files / 10,573,556,608 bytes, and 2 files / 5,283,759,068 bytes.
  Immutable metadata, six unique bounded safetensors headers, all nine package
  pipeline sources, and an upstream Apache-2.0 receipt are recorded without
  downloading full weights.

  Version 2.2 improves the execution contract: its combined text-to-image,
  image-to-image, and inpaint pipelines expose separate modern callbacks for
  the prior and decoder, so both denoising stages can be aborted. Admission is
  nevertheless closed because the connected loader still reuses the primary
  repository revision on the distinct prior repository, or downloads the
  prior's moving branch. The official depth-ControlNet and refiner snapshots
  contain only legacy pickle `.bin` weights; the refiner also names the 2.1
  pipeline class and has no model card or declared license. Backend two-stage
  assembly/bounds, safe auxiliary artifacts, output guardrails, snapshot
  license clarification, remote output review, and physical macOS evidence
  remain pending. No runtime, download, graph, capability, client, or media
  surface was added.
- [x] Admit Kandinsky 3 as a bounded single-stage Expert workflow. Backend
  `b85b073` and client `3007bf3` expose the exact public snapshot
  `kandinsky-community/kandinsky-3@bf79e6c219da8a94abb50235fdc4567eb8fb4632`
  for text-to-image and single-image editing through the generic Diffusers
  image facade. The repository is ungated, contains no Python, requires no
  remote code, and exposes seven fp16 safetensors files totaling
  28,390,829,958 bytes. Its canonical weight inventory, immutable metadata,
  pinned package pipeline/image-to-image/UNet source hashes, and upstream
  Apache-2.0 license receipt are sealed in
  `data/kandinsky-3-artifact-review.json`. The model snapshot declares
  Apache-2.0 in its card but contains no license file, so that clarification
  remains explicit.

  The backend-owned contract fixes both routes to the package's 1024px path,
  25 recommended steps, guidance 3, fp16 weights, at most 128 prompt tokens,
  and model or sequential CPU offload. Image editing accepts exactly one
  1,048,576-pixel source and uses the documented example strength 0.75. The
  package hardcodes its 128-token encoder path, so the generic sequence control
  is hidden while the backend still applies that adapter-specific default and
  bound. Both package routes expose the modern step callback used by MoDiff's
  cancellation contract. The package has no safety checker, and the resource
  envelope remains estimate-only, so Auto and Gallery are disabled and live
  execution remains unqualified.

  The expanded Transformers/Diffusers symbol surface passed locked clean-base
  install validation, fresh-process activation, the finite CLIP+PEFT workload,
  rollback, and a second clean-base process on Linux x86-64 at profile digest
  `sha256:b490f3012dbf1b01e400dc5284af1630b0fea738ce92643a7c8fe3dca0e4caca`.
  The bounded 1,554-byte evidence has SHA-256
  `d05812d8c95bd1f0f0ef770d117ff7da89c10e0c1e9a15946901ca8f2e36bb61`
  and retained no managed state. Two canonical graphs bring the deterministic
  catalog to 123 supported workflows. The complete backend overlay passes
  1,577 tests, 3,443 subtests, and three platform skips; Ruff E9/F,
  deterministic workflow verification, and the complete client gate also
  pass. The intentional client surface measures 530,549 compressed JavaScript
  bytes under the 531,456-byte ceiling. The exact app-managed snapshot download
  was accepted only after live free-space and aggregate queue-reservation
  preflight and remains queued/in progress; no older cache snapshot was
  removed. Remote real-weight memory/output safety/quality review,
  model-snapshot license-file clarification, and physical macOS execution
  remain pending independently, and no media has been generated.
- [x] Evaluate Kolors and keep its custom model license fail-closed. Backend
  `3788fe8` seals the exact public
  `Kwai-Kolors/Kolors-diffusers@7e091c75199e910a26cd1b51ed52c28de5db3711`
  snapshot. It is ungated, contains no repository Python, requires no remote
  code, and exposes a five-file / 17,813,668,046-byte fp16 safetensors
  partition. The canonical inventory, immutable metadata, and pinned
  package-owned Kolors text-to-image, image-to-image, ChatGLM encoder,
  tokenizer, and output source hashes are sealed in
  `data/kolors-artifact-review.json` without fetching weight bytes.

  The source review records both 1024px routes, the package's 50-step,
  guidance-5, 256-token defaults, image-edit strength 0.3, modern callbacks,
  interrupt flag, and CPU-offload sequence. It also records the missing
  package bounds for step count, input pixels, and output pixels and the absent
  safety checker. Those gaps require backend bounds and remote output review,
  but they are not the primary admission blocker.

  The immutable model card carries an Apache-2.0 tag and describes the code as
  Apache-2.0, while the same snapshot contains a distinct 14,920-byte
  `MODEL_LICENSE`. That model agreement purports to take effect on use or
  access, requires source/license and enforceable restriction propagation,
  prohibits using the model or its outputs to improve other large models, and
  requires separate authorization for cloud vendors or licensees over 100
  million monthly users. The README separately requests commercial
  registration. No task-scoped product acceptance, commercial registration,
  or legal approval was supplied, so no runtime/download catalog, capability,
  graph, client, Auto, or Gallery surface was added. In particular, the model
  was not submitted to the app download queue. The five focused review tests
  and complete 1,582-test backend overlay with 3,443 subtests and three
  platform skips pass; remote heavy-hardware review and physical macOS
  execution remain pending independently.
- [x] Evaluate classic Latent Diffusion and keep legacy serialization and
  cancellation gaps fail-closed. Backend `79db3b8` seals the exact public
  `CompVis/ldm-text2im-large-256@30de525ca11a880baea4962827fb6cb0bb268955`
  snapshot, its three-file / 6,152,286,891-byte required weight inventory,
  immutable metadata hashes, and pinned package-owned pipeline/encoder source
  identities in `data/latent-diffusion-artifact-review.json` without fetching
  any model weight bytes.

  All three required model components are legacy PyTorch pickle `.bin`
  artifacts, and the immutable repository provides no safetensors partition.
  The static Hub scanner's typical-Torch import result does not make executable
  pickle deserialization admissible under MoDiff's managed artifact policy.
  The package pipeline also exposes neither a denoising-step callback nor an
  interrupt flag, has no safety checker, and does not bound steps, output sides,
  or output pixels. The model card declares Apache-2.0 but the exact snapshot
  contains no license file. Consequently no runtime/download catalog,
  capability, graph, client, Auto, or Gallery surface was added, and the model
  was not submitted to the app download queue. Five focused review tests pass;
  safe official artifacts, cooperative cancellation, backend-owned resource
  bounds, immutable license receipt, remote execution/output review, and
  physical macOS evidence remain independent gates.
- [x] Evaluate LEDITS++ as a package transform over the already-managed Stable
  Diffusion bases and keep its multi-stage lifecycle fail-closed. Backend
  `7bddd15` binds the package-owned SD 1.5 and SDXL edit classes to MoDiff's
  existing exact base revisions, their already-recorded OpenRAIL terms, and
  the pinned Diffusers source hashes in `data/ledits-pp-source-review.json`.
  LEDITS++ is not a distinct model artifact, so it requires zero new weight
  bytes and no family-specific app download was submitted.

  Both routes require an inversion call followed by a separate edit call. The
  edit loop exposes a modern callback, but the mandatory inversion loop exposes
  neither a callback nor an interrupt flag, so the complete job cannot satisfy
  MoDiff's cooperative cancellation contract. Inversion also stores request
  latents/noise state on the pipeline instance; a shared generic facade would
  need an explicit isolation and lifecycle contract. The package does not
  bound inversion steps, image count, prompt count, dimensions, or input
  pixels; the SDXL route has no safety checker; and current package
  documentation warns that perfect inversion is no longer guaranteed. No
  runtime/download catalog, capability, graph, client, Auto, or Gallery surface
  was added. Five focused review tests pass; cancellation, request isolation,
  backend-owned bounds, generic multi-prompt editing, SDXL output guardrails,
  remote output review, and physical macOS evidence remain independent gates.
- [x] Admit LongCat Image generation and single-image editing without claiming
  live execution. Backend `212997c` and client `d1e5d7e` expose the exact
  public snapshots
  `meituan-longcat/LongCat-Image@d2ea50b79a930074c37b9b97ce45e3b2ea8cf4d8`
  and
  `meituan-longcat/LongCat-Image-Edit@7b54ef423aa7854be7861600024be5c56ab7875a`.
  Each selected partition contains seven safetensors files and approximately
  29.29 GB of weights; the immutable identities, repository metadata, pinned
  package pipeline/transformer/output sources, Transformers symbol contract,
  and upstream Apache-2.0 receipt are sealed in
  `data/longcat-image-artifact-review.json`. The model cards declare
  Apache-2.0 while both exact model snapshots omit the license file, so
  snapshot-specific clarification remains explicit.

  The two Expert-only routes use distinct immutable transformers while sharing
  the text encoder and VAE identities. Generation is bounded to 512-2048px
  sides in 16px increments, at most 1,048,576 output pixels and 50 steps, and
  disables the package's autoregressive prompt rewrite. Editing accepts exactly
  one at-most-1,048,576-pixel source between 1:4 and 4:1 aspect ratio and caps
  the package-derived rounded output at 1,088,000 pixels and 50 steps. Both
  package loops expose an interrupt flag checked per denoising step, but no
  safety checker; Auto and Gallery therefore remain disabled pending remote
  output review.

  The expanded optional-runtime surface passed clean-base locked installation,
  validation, fresh-process activation, a finite CLIP+PEFT workload, rollback,
  and clean restoration on Linux x86-64 at profile digest
  `sha256:76e4f0c1e4389bcefa0958c813ab1cbf3e745051b7435ec50df4d374f327520b`.
  The bounded 1,553-byte evidence has SHA-256
  `e2a73cd9fc7320ac84bb12a31efd3277ea3d3474591fed2a2dad2d7f787214fe`
  and retained no managed state. Two canonical graphs bring the deterministic
  catalog to 125 supported workflows. The complete backend overlay passes
  1,598 tests, 3,469 subtests, and three platform skips; Ruff E9/F,
  deterministic workflow verification, and the complete client gate also
  pass. The intentional client surface measures 530,610 compressed JavaScript
  bytes under the 531,456-byte ceiling.

  Both exact app-managed downloads were accepted only after per-snapshot and
  aggregate queue-reservation checks: 643,626,504,192 free bytes covered the
  existing 438,331,905,395-byte queue reservation, both new reservations of
  29,327,729,635 and 29,322,429,940 bytes, and the 68,719,476,736-byte safety
  reserve with 77,924,962,486 bytes of headroom. They remain queued/in progress;
  no older model was deleted and no media has been generated. Remote real-weight
  memory/output safety/quality review, model-snapshot license-file
  clarification, and physical macOS execution remain pending independently.
- [x] Admit Lumina Next and Lumina Image 2.0 text-to-image without claiming
  live execution. Backend `faad48b` and client `5b2f3db` expose the exact
  public snapshots
  `Alpha-VLLM/Lumina-Next-SFT-diffusers@0ee5ec90043acf5cb41fe96274af36eb7fad8d95`
  and
  `Alpha-VLLM/Lumina-Image-2.0@53504abd8178b30685b6c4c7a4cd181ff78b73e9`.
  Their four-file / 8,856,869,956-byte and six-file / 21,231,830,092-byte
  safetensors partitions, immutable repository/config identities, package
  pipeline and transformer hashes, Gemma symbol contract, upstream MIT and
  Apache-2.0 receipts, and estimate-only resource envelopes are sealed in
  `data/lumina-image-artifact-review.json`. Both model cards declare
  Apache-2.0 while their exact model snapshots omit a license file, so
  snapshot-specific clarification remains explicit.

  Both Expert-only routes are bounded to 512-2048px sides in 16px increments,
  at most 1,048,576 output pixels, guidance 4, and 256 prompt tokens. Lumina
  Next defaults to 30 steps, caps at 50, and disables optional caption
  cleaning. Lumina 2.0 uses its official 50-step recipe with CFG truncation
  fixed to 0.25 and normalization enabled. Its app/download contract contains
  an exact 17-file allowlist that excludes the two root `.pth` pickle artifacts
  and the demo asset. Both denoising loops retain the generic per-step callback,
  but neither package has a safety checker; Auto and Gallery remain disabled.

  The expanded optional-runtime surface passed clean-base locked installation,
  validation, fresh-process activation, a finite CLIP+PEFT workload, rollback,
  and clean restoration on Linux x86-64 at profile digest
  `sha256:db3485f5c9293e1cb76aac5128dd87f0089f091a9075fdd09fe541ae4132dff0`.
  The bounded 1,552-byte evidence has SHA-256
  `049a0563a60756a06397699ec2531433aed18850e70613f3e7969222b67d6875`
  and retained no managed state. Two canonical graphs bring the deterministic
  catalog to 127 supported workflows. The complete backend overlay passes
  1,604 tests, 3,495 subtests, and three platform skips; Ruff E9/F,
  deterministic workflow verification, and the complete client gate also
  pass. The intentional client surface measures 530,677 compressed JavaScript
  bytes under the 531,456-byte ceiling.

  Both exact downloads were submitted through the app only after aggregate
  reservation preflight. At submission, 625,606,197,248 free bytes covered the
  existing 487,559,232,373-byte queue reservation, the new 8,878,690,722-byte
  and 21,253,711,141-byte reservations, and the 68,719,476,736-byte safety
  reserve with 39,195,086,276 bytes of headroom. Fresh exact app plans on
  2026-08-14 now report all 16 Lumina Next files / 8,878,690,722 selected bytes
  and all 17 Lumina Image 2.0 files / 21,253,711,141 selected bytes complete at
  their immutable revisions. Both cache entries are installed, complete, and
  repair-free. No older model was deleted and no media has been generated.
  Remote real-weight memory/output safety/quality review, model-snapshot
  license-file clarification, and physical macOS execution remain pending
  independently.
- [x] Admit OmniGen v1 text generation, single-image editing, and ordered
  multi-reference editing without claiming live execution. Backend `11d5c16`
  and client `86500ec` expose the exact public snapshot
  `Shitao/OmniGen-v1-diffusers@016e2f61d12a98303f6bbdf122687694d7984268`.
  Its two-file / 8,085,311,084-byte safetensors partition, complete
  8,088,956,424-byte snapshot plan, immutable repository/config identities,
  pinned package pipeline/processor/transformer hashes, Llama tokenizer symbol
  contract, upstream MIT receipt, and estimate-only resource envelope are
  sealed in `data/omnigen-artifact-review.json`. The model card declares MIT
  while the exact model snapshot omits a license file, so snapshot-specific
  clarification remains explicit.

  All three Expert-only routes are bounded to 512-2048px sides in 16px
  increments, at most 1,048,576 output pixels, 50 steps, and text guidance
  2.5. Conditioned routes use image guidance 1.6 and accept at most three
  references totaling 3,145,728 pixels; the package independently preprocesses
  each input to a maximum 1024px side. MoDiff retains references as an ordered
  list, generates the package's continuous one-based
  `<img><|image_N|></img>` placeholders, rejects user-supplied reserved
  placeholder syntax, and maps the generic image-guidance field to the
  package's `img_guidance_scale` argument. The denoising loop retains the
  generic per-step callback but does not read its declared interrupt flag, and
  the package has no safety checker; Auto and Gallery remain disabled.

  The expanded optional-runtime surface passed clean-base locked installation,
  validation, fresh-process activation, a finite CLIP+PEFT workload, rollback,
  and clean restoration on Linux x86-64 at profile digest
  `sha256:7717741eb6fed3c8fbb9645fe13fd867d0e58ad06b06fe9009187c38ae840d42`.
  The bounded 1,553-byte evidence has SHA-256
  `39f6fab8538aec4e6a1aadd07edb2485d69f2e876490e44111e1f786475dfce2`
  and retained no managed state. Three canonical graphs bring the deterministic
  catalog to 130 supported workflows. The complete backend overlay passes
  1,610 tests, 3,525 subtests, and three platform skips; Ruff E9/F,
  deterministic workflow verification, and the complete client gate also
  pass. The intentional client surface measures 530,729 compressed JavaScript
  bytes under the 531,456-byte ceiling.

  The exact snapshot was submitted through the app only after aggregate
  reservation preflight. At submission, 612,865,241,088 free bytes covered the
  existing 495,504,680,091-byte queue reservation, the new
  8,088,956,424-byte reservation, and the 68,719,476,736-byte safety reserve
  with 40,552,127,837 bytes of headroom. A fresh exact app plan on 2026-08-14
  now reports all 11 files / 8,088,956,424 selected bytes complete at the
  immutable revision; its cache entry is installed, complete, and repair-free.
  No older model was deleted and no media has been generated. Remote
  real-weight memory/output safety/quality review, model-snapshot license-file
  clarification, and physical macOS execution remain pending independently.
- [x] Admit Ovis Image 7B text-to-image without claiming live execution.
  Backend `df2fa98` and client `ab47b67` expose the exact public Apache-2.0
  snapshot
  `ATH-MaaS/Ovis-Image-7B@41be1c5821a92c970d63d7eb595a2fd3fe32b22e`.
  Its selected 21-file Diffusers snapshot contains five safetensors files /
  21,790,968,814 weight bytes and totals 21,806,937,901 download bytes. The
  allowlist includes the immutable LICENSE and NOTICE while excluding both
  duplicate root-native checkpoints and the complete bundled `Ovis2.5-2B/`
  subtree, including its repository Python. Exact weight identities,
  repository/config hashes, pinned package pipeline/transformer hashes, Qwen
  runtime symbols, terms receipts, and estimate-only resource envelope are
  sealed in `data/ovis-image-artifact-review.json`; remote-code trust is never
  enabled.

  The Expert-only route is bounded to 512-2048px sides in 16px increments, at
  most 1,048,576 output pixels, 50 steps, guidance 5, and 256 prompt tokens.
  It retains the package's per-step callback, denoising-loop interrupt check,
  negative prompt, and text-encoder/transformer/VAE CPU-offload sequence. The
  package has no safety checker, so Auto and Gallery remain disabled pending
  remote output review.

  The expanded optional-runtime surface passed clean-base locked installation,
  validation, fresh-process activation, a finite CLIP+PEFT workload, rollback,
  and clean restoration on Linux x86-64 at profile digest
  `sha256:64d2b473aa9d8dae0472d354468e05982f0f78f23ccb2cc7e7e335914f53c3ce`.
  The bounded 1,554-byte evidence has SHA-256
  `c4ee928c7be0509c1e49507e8d3b93a959f1a99bfcd75bdef070d8c9eaee14e8`
  and retained no managed state. One canonical graph brings the deterministic
  catalog to 131 supported workflows. The complete backend overlay passes
  1,616 tests, 3,542 subtests, and three platform skips; Ruff E9/F, package
  compatibility, portable preflight, shell syntax, deterministic workflow
  verification, and the complete client check also pass. The intentional
  client surface measures 530,754 compressed JavaScript bytes under the
  531,456-byte ceiling.

  The exact safe selection was submitted through the app only after aggregate
  reservation preflight. At submission, 590,321,881,088 free bytes covered the
  existing 489,903,040,881-byte queue reservation, the new
  21,806,937,901-byte reservation, and the 68,719,476,736-byte safety reserve
  with 9,892,425,570 bytes of headroom. A fresh exact app plan on 2026-08-14
  now reports all 21 files / 21,806,937,901 selected bytes complete at the
  immutable revision; its cache entry is installed, complete, and repair-free.
  No older model was deleted and no media has been generated. Remote
  real-weight memory/output safety/quality review and physical macOS execution
  remain pending independently.
- [x] Admit PRX 512 SFT text-to-image without claiming live execution. Backend
  `012b101` and client `9c320a4` expose the exact public snapshot
  `Photoroom/prx-512-t2i-sft@2996423bc26e8eaca48774fac1797484214dfea0`.
  Its complete 19-file / 15,514,188,109-byte app selection contains five
  safetensors files / 15,475,492,212 weight bytes and no repository Python.
  Exact weight identities, immutable repository/config hashes, pinned package
  pipeline/transformer hashes, and an estimate-only resource envelope are
  sealed in `data/prx-artifact-review.json`; remote-code trust is never
  enabled. The bundled Apache-2.0 LICENSE and NOTICE are retained, and the
  NOTICE's incorporated T5-Gemma terms and prohibited-use policy remain
  explicit in both the artifact receipt and the Expert surface.

  The Expert-only route uses the native 512px SFT recipe and is bounded to the
  package's 352-704px aspect bins in 32px increments, at most 262,144 output
  pixels, 28 steps, guidance 5, and 256 prompt tokens. The generic Studio token
  limit maps explicitly to PRX's `tokenizer_max_length` argument. The package
  exposes a per-step callback but no denoising-loop interrupt flag or safety
  checker, so stop requests fail closed by raising at a step boundary while
  Auto and Gallery remain disabled pending remote output review.

  The expanded PRX/T5-Gemma optional-runtime surface passed clean-base locked
  installation, validation, fresh-process activation, a finite CLIP+PEFT
  workload, rollback, and clean restoration on Linux x86-64 at profile digest
  `sha256:1705482bef0b94433b5380f71c0ed9a1e6ccb97427e1b8b4bb9238ad1547d0e3`.
  The bounded 1,553-byte evidence has SHA-256
  `996009d1a2b09be13115a637b8fffa390e43679ece8d80302e9d5e77b4e4c34c`
  and retained no managed state. One canonical graph brings the deterministic
  catalog to 132 supported workflows. The complete backend overlay passes
  1,621 tests, 3,559 subtests, and three platform skips; Ruff E9/F, package
  compatibility, portable preflight, shell syntax, deterministic workflow
  verification, and the complete client check also pass. The intentional
  client surface measures 530,791 compressed JavaScript bytes under the
  531,456-byte ceiling.

  The exact snapshot was submitted through the app only after aggregate
  reservation preflight. At submission, 567,349,919,744 free bytes covered the
  existing 474,119,351,599-byte queue reservation, the new
  15,514,188,109-byte reservation, and the 68,719,476,736-byte safety reserve
  with 8,996,903,300 bytes of headroom. A fresh exact app plan on 2026-08-14
  now reports all 19 files / 15,514,188,109 selected bytes complete at the
  immutable revision; its cache entry is installed, complete, and repair-free.
  No older model was deleted and no media has been generated. Remote
  real-weight memory/output safety/quality review and physical macOS execution
  remain pending independently.
- [x] Admit Nucleus Image 17B MoE text-to-image without claiming live
  execution. Backend `86cc756` and client `8b4f002` expose the exact public
  snapshot
  `NucleusAI/Nucleus-Image@5e963db4fd0a65c7e4faf53ca2d4eca567c4dcfa`.
  Its complete 38-file / 51,656,728,957-byte Python-free snapshot contains 12
  safetensors files / 51,633,577,862 weight bytes. Exact weight identities,
  immutable repository/config hashes, pinned package pipeline/transformer
  hashes, Qwen3-VL runtime symbols, and an estimate-only resource envelope are
  sealed in `data/nucleus-image-artifact-review.json`; remote-code trust is
  never enabled. The model card declares Apache-2.0, but the exact immutable
  snapshot contains no LICENSE or NOTICE file, so owner clarification remains
  a separate gate.

  The Expert-only route uses the native 1024px, 50-step, guidance-4 recipe and
  admits all seven official aspect buckets through 768-1344px sides in 32px
  increments, at most 1,060,864 output pixels, and 1024 prompt tokens. It
  retains the package's per-step callback, denoising-loop interrupt check,
  negative prompt, and text-encoder/transformer/VAE CPU-offload sequence. This
  is a base checkpoint without post-training or a safety checker, so Auto and
  Gallery remain disabled pending remote output and policy review.

  The expanded Nucleus/Qwen3-VL optional-runtime surface passed clean-base
  locked installation, validation, fresh-process activation, a finite
  CLIP+PEFT workload, rollback, and clean restoration on Linux x86-64 at
  profile digest
  `sha256:1cf278aa4e690212b0a50b6b1e30ad77e86fb9af6fc13dbb89b37139a35c1c45`.
  The bounded 1,554-byte evidence has SHA-256
  `5213c07a29232ed6eef753a6114ed5667f03d38b7a1eabb1443d89204ee1a349`
  and retained no managed state. One canonical graph brings the deterministic
  catalog to 133 supported workflows. The complete backend overlay passes
  1,626 tests, 3,576 subtests, and three platform skips; Ruff E9/F, package
  compatibility, portable preflight, shell syntax, deterministic workflow
  verification, and the complete client check also pass. The intentional
  client surface measures 530,833 compressed JavaScript bytes under the
  531,456-byte ceiling.

  The exact snapshot was not submitted after the required app-only aggregate
  reservation preflight. At the latest check, 535,900,401,664 free bytes minus
  the existing 451,832,786,377-byte queue reservation and the
  68,719,476,736-byte safety reserve left 15,348,138,551 bytes before this
  snapshot, making the new reservation short by 36,308,590,406 bytes. No
  direct weight download was performed and no older model was deleted. Owner
  license clarification, app storage capacity, remote real-weight
  memory/output safety/quality review, and physical macOS execution remain
  pending independently.
- [x] Admit the remaining official Wan 2.1 14B Modular-compatible repository
  variants without claiming live execution. Backend `e4c2385` adds exact
  repository-scoped Models Loader aliases for T2V-14B at
  `38ec498cb3208fb688890f8cc7e94ede2cbd7f68` and I2V-14B-720P at
  `eb849f76dfa246545b65774a9e25943ee69b3fa3`, alongside the already reviewed
  I2V-14B-480P and FLF-14B-720P snapshots. The four exact safetensors surfaces
  are respectively 18 files / 80,385,341,396 bytes, 21 files /
  90,075,953,948 bytes for each I2V variant, and 21 files /
  90,077,762,204 bytes for FLF. Canonical inventory digests, all transformer
  shard hashes, standard index/config hashes, immutable revisions, source
  recipes, and estimate-only resource bounds are sealed without downloading
  weights.

  The admission remains generic and fail-closed: T2V-14B must resolve from
  `WanPipeline` to the pinned `WanModularPipeline`; both I2V repositories must
  resolve from `WanImageToVideoPipeline` to the pinned
  `WanImage2VideoModularPipeline`; and FLF retains its distinct processor,
  positional-embedding, and last-image contract. I2V-480P and I2V-720P can
  satisfy only image-to-video routing, while the FLF artifact can satisfy only
  first/last-frame routing. Wrong repository, workflow, revision, component
  type, or index class fails before block initialization. No high-level mode,
  client model-name branch, Auto path, template, Gallery asset, generated
  media, or live qualification claim was added. The clean optional overlay
  passes the 100-test focused matrix with 175 subtests; remote heavy-hardware
  execution and physical macOS evidence remain pending independently.
- [ ] Evaluate other heavy video families.
  - [x] **Wan Animate 2 pinned Modular contract closure:** backend `d31d5b6`
    and client `fc67a7f` register the two package exports that were absent from
    the reviewed contract snapshot at Diffusers
    `bb56997d4b7e87f0743f26a612f49ec4e7ce7213`:
    `WanAnimate2ModularPipeline` and
    `WanAnimate2DistilledModularPipeline`. Both are Expert-visible,
    contract-only video records with one generic `character_animate` workflow,
    required prompt/image/driving-video inputs, video output, and distinct base
    versus distilled denoise steps. The generated truth now seals all 33
    exported Modular classes and 93 workflows; a pinned-overlay regression
    requires the executable and contract-only registries to equal that exact
    exported-class set.

    The snapshot preserves the actual composed schemas: although the upstream
    distilled prose describes ten steps, both pinned classes currently publish
    a 40-step default. No default repository, artifact admission, runnable
    mode, Auto/template/Gallery surface, or live qualification was inferred.
    The focused optional-overlay gate passes 45 tests and 664 subtests; the
    complete overlay passes 1,668 tests and 3,591 subtests with three platform
    skips, and the clean base passes 1,631 tests and 3,279 subtests with 40
    optional-runtime skips. Ruff, package compatibility, snapshot verification,
    the complete client check, the 533,100 / 533,504-byte gzip budget, and the
    107-case mocked Studio browser sweep pass. No weights, models, or media were
    downloaded, deleted, or generated for this slice. Artifact admission,
    real-weight remote execution, output review, and physical macOS evidence
    remain pending independently.
  - [x] **Wan Animate 2 immutable artifact/source review:** backend `2b86e63`
    seals the public, Python-free base
    `Wan-AI/Wan2.2-Animate-2-14B-Diffusers@7d48412d7b903ff3a89f4f5a960d99e1899605a1`
    and distilled
    `Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers@59e4141466bcb1bf9733eca1bc78be6891c9fbdf`
    snapshots. Each has 31 files and nine BF16 safetensors weights totaling
    45,920,934,868 bytes; five shared image/text/VAE files account for
    13,131,039,324 bytes, while the four 32,789,895,544-byte transformer
    shards have distinct exact hashes. Both cards declare Apache-2.0 only in
    metadata, include no license file, and provide no detailed usage or safety
    guidance.

    Admission remains contract-only for exact source reasons. Both standard
    indexes name `WanAnimate2Pipeline`, which the pinned package does not
    export. The matching Modular indexes name the reviewed base/distilled
    classes but leave four component revisions null and point both scheduler
    and transformer at mutable `refs/pr/2` or `refs/pr/1` references. The
    package generation path additionally requires compiled flex attention at
    video resolution, carries decoded tail frames across 81-frame segments,
    and retains the unresolved distilled ten-step prose versus 40-step composed
    schema mismatch. No runtime/download catalog, repository default, runnable
    mode, or client change was added. The focused gate passes 13 tests, one
    optional-runtime skip, and 44 subtests; the complete clean-base backend gate
    passes 1,636 tests, 40 skips, and 3,279 subtests with Ruff, 66-package
    compatibility, shell/diff checks, and portable preflight green. No model
    weights or media were downloaded, deleted, or generated. Immutable
    component normalization, remote compiled execution/output review, and
    physical macOS evidence remain pending.
- [x] **P6.61 Bound AuraFlow app download selection:** backend `ed3982a`
  binds `fal/AuraFlow-v0.3@2cd8588f04c886002be4571697d84654a50e3af3`
  to the exact 18-file runnable fp16 selection used by the admitted
  `AuraFlowPipeline`. The selection contains the four reviewed safetensors
  weights, their exact variant index, package configs/tokenizer metadata, and
  immutable license/card receipts. It excludes the duplicate native single
  file, default text encoder, three-shard transformer, default VAE, and ComfyUI
  workflow surfaces.

  A real app `GET /hf_download/plan` over that explicit selection reports
  18 files / 16,837,479,394 bytes instead of the repository-wide 26 files /
  65,964,186,183 bytes. Plan and POST regressions require the same capability
  allowlist, and the artifact regression requires every reviewed fp16 weight
  while excluding each duplicate weight surface. The focused app/artifact
  matrix passes 88 tests and 46 subtests; pinned Ruff E9/F and the 66-package
  compatibility check pass. The already-running repository-wide app task was
  not interrupted, replaced, or deleted, and no second AuraFlow POST was made;
  this bound governs future installs and repairs after the app restarts.
- [x] **P6.62 Bound Chroma app download selection:** backend `0922243`
  binds `lodestones/Chroma1-HD@0e0c60ece1e82b17cb7f77342d765ba5024c40c0`
  to its exact 18-file runnable Diffusers selection. The allowlist contains the
  five reviewed safetensors weights, both shard indexes, component configs,
  tokenizer, and immutable card receipt. It excludes the duplicate
  17,800,038,288-byte native single-file checkpoint plus the ComfyUI workflow
  and demo images.

  A real app plan reports the selected 18 files / 27,493,360,428 bytes are
  already complete, versus 45,300,383,271 bytes for the repository-wide tree.
  Plan and POST regressions require the identical selection and the artifact
  regression binds it to every reviewed component weight. The focused
  app/artifact matrix passes 95 tests and 46 subtests; pinned Ruff E9/F and the
  66-package compatibility check pass. No new POST was necessary, and the
  older complete repository snapshot and its duplicate bytes were preserved.
- [x] **P6.63 Bound Allegro, Latte, and Mochi app download selections:** backend
  `cb3448d` binds all three admitted video routes to exact runnable Diffusers
  file allowlists. Allegro selects 18 files / 25,293,916,977 bytes and excludes
  both duplicate unsafe PyTorch text-encoder shards. Latte selects 18 files /
  23,615,823,652 bytes and excludes the legacy `.pt` checkpoint plus the
  unreferenced temporal decoder. Mochi selects 21 files / 40,025,271,759 bytes
  and preserves the indexed four-shard T5 plus BF16 transformer/VAE route while
  excluding the flat-format duplicate, unindexed two-shard T5, FP32
  transformer/VAE, and demo asset.

  Real app plans prove those selections instead of the repository-wide
  44,370,333,435-byte Allegro, 28,238,181,756-byte Latte, and
  133,509,491,072-byte Mochi trees. Both app planning and POST admission derive
  the same capability files; artifact tests require every reviewed weight and
  reject every recorded duplicate/unsafe surface. The focused matrix passes
  117 tests and 49 subtests with pinned Ruff E9/F and 66-package compatibility
  green. None fit beside the active app queue, so no POST was submitted and no
  existing complete or partial cache entry was removed.
- [x] **P6.64 Bound Stable Audio and Stable Video app download selections:**
  backend `8c96321` binds both existing admitted media routes to the runnable
  component sets selected by their loaders. Stable Audio selects 19 files /
  5,348,079,831 bytes, retaining its safetensors projection model, text
  encoder, transformer, VAE, tokenizer/config surface, license/card, and both
  dataset attribution receipts. It excludes the duplicate original `.ckpt`
  and single-file safetensors checkpoints, original model configs, and demo
  image from the 15,680,736,700-byte repository tree. Stable Video Diffusion
  selects 12 files / 4,509,218,296 bytes, matching the loader's exact `fp16`
  variant for its image encoder, UNet, and VAE. It excludes all three default
  full-precision component weights, the duplicate single-file checkpoint, and
  demo image from the 18,313,865,678-byte repository tree.

  App plan and POST regressions require those identical capability-derived
  selections, while the artifact regressions reject unsafe and duplicate
  surfaces. The focused media, capability, app, and loader matrix passes 270
  tests, two environment skips, and 544 subtests; pinned Ruff E9/F,
  66-package compatibility, and diff checks pass. Fresh exact app plans report
  400,730,861,568 free bytes, 328,468,945,405 queued reservation bytes, and the
  68,719,476,736-byte safety reserve, so neither additional request fits beside
  the active queue. No duplicate POST was submitted; the pre-existing
  repository-wide app transfers were not interrupted and no older or partial
  cache entry was deleted.
- [x] **P6.65 Bound AudioLDM2 and Shap-E app download selections:** backend
  `56faa30` makes Model Manager use the same safe component surfaces already
  enforced by both loaders. AudioLDM2 selects 28 files / 4,480,959,446 bytes,
  including all seven safetensors component weights and their configs,
  tokenizers, scheduler, feature extractor, model index, and card. It excludes
  all seven co-published legacy `.bin` duplicates, totaling 4,475,112,301
  bytes, from the 8,956,071,747-byte repository tree. Shap-E selects 14 files /
  1,332,951,857 bytes for the exact fp16 prior, CLIP text encoder, pre-rename
  safe renderer, tokenizer, scheduler, configs, index, and card. It excludes
  the three legacy component `.bin` files plus the renamed unsafe-only renderer
  directory, totaling 3,568,036,469 bytes, from the 4,900,988,326-byte tree.

  Plan and POST regressions bind both capability-derived allowlists, and focused
  artifact tests match them to their explicit loaders while rejecting every
  unsafe duplicate. The media, capability, app, and loader matrix passes 277
  tests, three optional-runtime skips, and 550 subtests; pinned Ruff E9/F,
  66-package compatibility, and diff checks pass. Fresh exact app plans report
  every selected byte already complete with zero remaining bytes, so no POST or
  deletion was necessary; the preserved full snapshots remain available for
  preview regression testing.
- [x] **P6.66 Bound Marigold app download and loader serialization:** backend
  `c841203` binds `prs-eth/marigold-depth-lcm-v1-0` to the exact 14-file /
  5,161,610,352-byte float32 safetensors component surface used by its generic
  prediction-map route. The loader now explicitly requires safe serialization
  instead of relying on the presence of preferred files. The app allowlist
  retains the default text encoder, UNet, VAE, tokenizer, scheduler, configs,
  index, and card while excluding six legacy `.bin` weights and three unused
  fp16 duplicate variants: 10,320,895,232 bytes from the
  15,482,505,584-byte repository tree.

  App plan and POST regressions bind the capability-derived selection, while a
  focused perception regression requires the loader's safe-serialization flag,
  exact default-variant weights, and complete exclusion of legacy pickle
  files. The capability, app, perception, and loader matrix passes 212 tests,
  three optional-runtime skips, and 867 subtests; pinned Ruff E9/F, 66-package
  compatibility, and diff checks pass. A fresh exact app plan reports every
  selected byte already complete with zero remaining bytes, so no POST or
  deletion occurred. The older extra variants and unfinished cache blobs were
  preserved for regression testing.
- [x] **P6.67 Bound Whisper Tiny app download selection:** backend `4783400`
  binds `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`
  to one exact 13-file / 155,455,649-byte Transformers runtime surface. The
  selection retains the safetensors model, processor/tokenizer data,
  generation/config metadata, and immutable card/attribute receipts while
  excluding the duplicate 151,095,027-byte PyTorch pickle, 151,048,591-byte
  Flax checkpoint, and 151,253,960-byte TensorFlow checkpoint from the
  608,853,227-byte repository tree. This matches the existing generic ASR
  loader's safetensors-only, no-remote-code boundary.

  App plan and POST regressions bind the capability-derived allowlist, and a
  focused speech artifact regression rejects every alternate framework and
  legacy pickle surface. The capability, app, and speech matrix passes 101
  tests and 80 subtests; pinned Ruff E9/F, 66-package compatibility, and diff
  checks pass. A fresh exact app plan found all runtime bytes present and only
  21,225 bytes of card/attribute receipts remaining; with
  390,247,436,288 free bytes, 309,318,466,805 queued reservation bytes, and the
  68,719,476,736-byte safety reserve, the bounded completion fit and was
  submitted through the app. It remains governed by the app's existing
  transfer queue; no older model or alternate cached framework file was
  deleted.
- [x] **P6.68 Bound unconditional-image app downloads and loader serialization:**
  backend `42fa625` makes both small-model routes safetensors-only in the loader
  and Model Manager. DDPM and DDIM share the exact six-file /
  143,025,496-byte `google/ddpm-cifar10-32` selection, excluding the
  143,101,489-byte legacy `.bin`, repository Python, and four demo images from
  the 286,139,260-byte tree. The consistency-model route selects six files /
  1,183,678,409 bytes from `openai/diffusers-cd_imagenet64_l2`, excluding its
  1,183,833,415-byte legacy `.bin` duplicate from the 2,367,511,824-byte tree.
  Both selections retain their safetensors model, scheduler/config surface,
  index, and immutable card/attribute receipts.

  App plan and POST regressions bind both capability-derived allowlists; the
  focused artifact regression requires the shared DDPM/DDIM identity, explicit
  safe-serialization flags, code-free selection, and complete legacy-weight
  exclusion. The capability, app, unconditional, and loader matrix passes 213
  tests, three optional-runtime skips, and 872 subtests; pinned Ruff E9/F,
  66-package compatibility, and diff checks pass. Independent fresh app plans
  found the runtime weights complete and only 4,277 plus 11,427 receipt bytes
  remaining; both exact fitting repairs were submitted concurrently through
  the app and remain governed by its existing bounded queue. No older model,
  unsafe duplicate, demo, or code file was deleted.
- [x] **P6.69 Bound LCM DreamShaper app download and loader serialization:**
  backend `08c34f4` binds
  `SimianLuo/LCM_Dreamshaper_v7@a85df6a8bd976cdd08b4fd8f3b73f229c9e54df5`
  to the exact 17-file / 5,482,979,343-byte Diffusers runtime surface. The
  selection retains the safety checker, text encoder, UNet, VAE, tokenizer,
  scheduler, feature-extractor/config files, index, and immutable
  card/attribute receipts. It excludes the duplicate root checkpoint, ONNX
  exports, repository Python, and demo images: 7,710,367,026 bytes from the
  13,193,346,369-byte repository tree. The generic image loader now explicitly
  requires safe serialization.

  App plan and POST regressions bind the capability-derived allowlist, while a
  focused latent-image artifact regression requires the exact component
  safetensors and rejects the duplicate checkpoint, ONNX, code, and demo
  surfaces. The capability, app, latent-image, and loader matrix passes 212
  tests, three optional-runtime skips, and 871 subtests; pinned Ruff E9/F,
  66-package compatibility, and diff checks pass. A fresh exact app plan found
  all runtime weights complete and only 5,070 receipt bytes remaining; with
  384,460,107,776 free bytes, 303,551,304,790 queued reservation bytes, and the
  68,719,476,736-byte safety reserve, the bounded completion fit and was
  submitted through the app. It remains governed by the existing transfer
  queue, and no older cached artifact was deleted.
- [x] **P6.70 Bound Sana 0.6B fp16 app download selection:** backend `982c1a3`
  binds
  `Efficient-Large-Model/Sana_600M_1024px_diffusers@28f3af7689de15f3883d5863059a2fca0aa9b829`
  to the exact 17-file / 7,700,017,758-byte fp16 Diffusers runtime surface used
  by its existing safetensors-only loader. The selection retains the fp16 text
  encoder shards/index, transformer and VAE weights, tokenizer, scheduler,
  configs, model index, license, card, and attribute receipts. It excludes the
  default text-encoder aliases plus the full-precision transformer and default
  VAE alias: 8,844,837,635 logical bytes from the 16,544,855,393-byte repository
  tree. Existing identical-content aliases remain in cache.

  App plan and POST regressions bind the capability-derived allowlist, and a
  focused Sana artifact regression matches the loader's exact fp16 variant and
  rejects every default alias or legacy serialization surface. The capability,
  app, Sana, and loader matrix passes 213 tests, three optional-runtime skips,
  and 872 subtests; pinned Ruff E9/F, 66-package compatibility, and diff checks
  pass. A fresh exact app plan reported zero remaining bytes, with
  382,775,398,400 free bytes, 302,075,919,129 queued reservation bytes, and the
  68,719,476,736-byte safety reserve intact, so no POST or deletion occurred.
- [x] **P6.71 Bound FLUX.2 Klein component download and loader serialization:**
  backend `80f7369` binds
  `black-forest-labs/FLUX.2-klein-4B@e7b7dc27f91deacad38e78976d1f2b499d76a294`
  to the exact 21-file / 15,980,152,900-byte Diffusers component surface used
  by its text, edit, multi-reference, inpaint, and outpaint routes. The
  selection retains the text encoder, transformer, VAE, tokenizer, scheduler,
  configs, model index, license, card, and attribute receipts while excluding
  the duplicate 7,751,105,712-byte native checkpoint and 8,748,835 bytes of
  demo images from the 23,740,007,447-byte repository tree. Both generic Klein
  loaders now explicitly require safe serialization.

  App plan and POST regressions bind the capability-derived allowlist; the
  focused FLUX artifact regression requires the exact component safetensors,
  both loader flags, and complete native-checkpoint/demo exclusion. The
  capability, app, FLUX, and loader matrix passes 213 tests, three
  optional-runtime skips, and 875 subtests; pinned Ruff E9/F, 66-package
  compatibility, and diff checks pass. A fresh exact app plan reported zero
  remaining bytes, with 379,585,880,064 free bytes, 295,904,744,282 queued
  reservation bytes, and the 68,719,476,736-byte safety reserve intact, so no
  POST or deletion occurred.
- [x] **P6.72 Bound primary FLUX.1 component downloads and loader
  serialization:** backend `274b158` binds schnell, dev, and Krea to the exact
  component surfaces consumed by the generic text/edit/inpaint loaders. The
  immutable selections contain 25 / 26 / 26 files and 33,725,928,351 /
  33,746,408,222 / 33,746,412,027 bytes respectively, retaining their text
  encoders, transformers, VAEs, tokenizers, schedulers, configs, indexes, and
  available rights receipts. They exclude the three 23.78 GB native
  checkpoints, three root autoencoders, and demo images: 72,409,924,695 logical
  bytes from the combined 173,628,673,295-byte repository trees. The shared
  FLUX text, image-to-image, and inpaint loaders now explicitly require safe
  serialization.

  App plan and POST regressions bind all three capability-derived allowlists;
  the focused FLUX artifact regression requires each exact component surface,
  the shared loader flags, and complete native-checkpoint/autoencoder/demo
  exclusion. The capability, app, FLUX, and loader matrix passes 213 tests,
  three optional-runtime skips, and 884 subtests; pinned Ruff E9/F, 66-package
  compatibility, and diff checks pass. Three concurrent fresh exact app plans
  reported zero remaining bytes, with 379,114,487,808 free bytes,
  295,904,744,282 queued reservation bytes, and the 68,719,476,736-byte safety
  reserve intact, so no POST or deletion occurred. All excluded cached files
  remain available for preview regression testing.
- [x] **P6.73 Bound conditioned FLUX.1 component downloads and loader
  serialization:** backend `d39bfe2` binds Depth, Canny, Fill, and Kontext to
  the exact component surfaces consumed by their generic control, inpaint,
  outpaint, edit, and multi-reference routes. Depth/Canny each retain 28 files
  and 43,685,183,091 / 43,685,182,867 bytes; Fill/Kontext each retain 26 files
  and 33,916,013,136 / 33,746,413,969 bytes. The allowlists preserve the text
  encoders, transformers, VAEs, tokenizers, schedulers, configs, indexes, and
  rights receipts while excluding the four native checkpoints, four root
  autoencoders, and Kontext teaser: 96,561,961,318 logical bytes from the
  combined 251,594,754,381-byte repository trees. The shared control, Fill,
  Kontext, and Kontext-inpaint loaders now explicitly require safe
  serialization.

  App plan and POST regressions bind all four capability-derived allowlists;
  the focused FLUX artifact regression requires each component surface, every
  loader flag, and complete native-checkpoint/autoencoder/demo exclusion. The
  capability, app, FLUX, server, and loader matrix passes 234 tests, three
  optional-runtime skips, and 925 subtests; pinned Ruff E9/F, 66-package
  compatibility, and diff checks pass. Four concurrent fresh exact app plans
  reported zero remaining bytes, with 374,784,069,632 free bytes,
  295,904,744,282 queued reservation bytes, and the 68,719,476,736-byte safety
  reserve intact, so no POST or deletion occurred. All excluded cached files
  remain available for preview regression testing.
- [x] **P6.74 Bound the shared SD1.5 image/video download union and image-loader
  serialization:** backend `896a723` binds the common immutable SD1.5 base to
  the exact 21-file / 8,223,292,159-byte union required by both admitted loader
  families. Generic image, edit, inpaint, ControlNet, and PAG routes use the
  float32 safetensors components, while AnimateDiff and AnimateLCM explicitly
  use the matching fp16 variants; both are retained alongside shared
  tokenizer, scheduler, feature-extractor, configs, model index, card, and
  attribute receipts. The allowlist excludes legacy pickle weights, the unused
  non-EMA UNet, single-file checkpoints, and inference YAML: 39,036,647,490
  bytes from the 47,259,939,649-byte repository tree. All five generic image
  adapters now explicitly require safe serialization; the two video loaders
  already did so.

  App plan and POST regressions bind every SD1.5-backed capability to one
  identical allowlist, while a focused shared-artifact regression requires
  both precision variants and rejects all pickle, checkpoint, non-EMA, and
  YAML surfaces. The capability, app, SD1.5, image-loader, and video-loader
  matrix passes 325 tests, five optional-runtime skips, and 1,126 subtests;
  pinned Ruff E9/F, 66-package compatibility, and diff checks pass. A fresh
  exact app plan found the float32 files complete and 2,740,639,959 fp16 bytes
  remaining; with 374,465,064,960 free bytes, 295,904,744,282 queued reservation
  bytes, and the 68,719,476,736-byte safety reserve, the repair fit and was
  submitted through the app. It remains governed by the existing transfer
  queue; no older SD1.5 artifact was deleted.
- [x] **P6.75 Bound Z-Image component download and loader serialization:**
  backend `8f96945` binds
  `Tongyi-MAI/Z-Image-Turbo@f332072aa78be7aecdf3ee76d5c247082da564a6`
  to the exact 21-file / 32,848,321,404-byte Diffusers component surface used
  by its text-to-image, image-to-image, and inpaint-capable generic adapters.
  The allowlist retains the text encoder, transformer, VAE, tokenizer,
  scheduler, configs, indexes, card, and attribute receipt while excluding the
  gallery PDF and ten documentation/showcase images: 51,345,993 bytes from the
  32,899,667,397-byte repository tree. All three Z-Image adapters now
  explicitly require safe serialization.

  App plan and POST regressions bind the server capability to the exact
  allowlist; a focused Z-Image artifact regression requires all sharded
  safetensors indexes, excludes every asset and legacy serialization surface,
  and verifies the three loader flags. The capability, app, Z-Image, server,
  and loader matrix passes 232 tests, three optional-runtime skips, and 914
  subtests; pinned Ruff E9/F, 66-package compatibility, and diff checks pass. A
  fresh exact app plan reported zero remaining bytes, with 370,478,370,816 free
  bytes, 298,645,384,241 queued reservation bytes, and the 68,719,476,736-byte
  safety reserve intact, so no POST or deletion occurred. The excluded cached
  assets remain available for preview regression testing.
- [x] **P6.76 Bound PixArt Sigma component download selection:** backend
  `a0fe20c` binds
  `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS@e102b3591cc82e97071b8b4cb90d834d0c487207`
  to the exact 15-file / 21,828,231,839-byte Diffusers component surface used
  by its existing safetensors-only generic image loader. The allowlist retains
  the text encoder, transformer, VAE, tokenizer, scheduler, configs, indexes,
  card, and attribute receipt while excluding the three documentation images:
  4,258,550 bytes from the 21,832,490,389-byte repository tree.

  App plan and POST regressions bind the capability-derived selection; a
  focused PixArt artifact regression requires the exact component weights,
  existing safe-serialization flag, and complete documentation-asset
  exclusion. The capability, app, PixArt, and loader matrix passes 212 tests,
  three optional-runtime skips, and 883 subtests; pinned Ruff E9/F, 66-package
  compatibility, and diff checks pass. A fresh exact app plan reported zero
  remaining bytes. Current free bytes were 366,990,446,592 against
  298,645,384,241 queued reservation bytes and the 68,719,476,736-byte safety
  reserve, so the aggregate queue envelope was temporarily 374,414,385 bytes
  over the limit while an existing transfer finalized. PixArt required no new
  bytes; no POST, interruption, or deletion occurred, and the cached
  documentation assets remain preserved.
- [x] **P6.77 Bound admitted Wan video component downloads and loader
  serialization:** backend `e3fc376` binds the four currently cached admitted
  Wan families to their exact Diffusers runtime surfaces. Wan 2.1 T2V/video
  edit retains 21 files / 28,928,908,149 bytes; Wan 2.1 VACE retains 19 files /
  19,037,151,627 bytes; Wan 2.2 dual-expert I2V retains 43 files /
  126,202,571,522 bytes; and Wan 2.2 TI2V retains 22 files / 34,201,437,893
  bytes. Each allowlist includes the required text encoder, transformer(s),
  VAE, tokenizer, scheduler, configs, indexes, card, and attribute receipt while
  excluding repository documentation and example media: 15,892,213 bytes from
  the combined 208,385,961,404-byte trees. VACE, T2V, video-to-video, TI2V, and
  dual-expert I2V loads now all pass `use_safetensors=True` to both explicit
  VAE and pipeline component loads.

  App plan and POST regressions bind all four capabilities to their exact
  allowlists. Focused artifact and loader regressions require every indexed
  component, exclude all asset/example and legacy serialization surfaces, and
  exercise safetensors propagation through each distinct Wan loader. The Wan,
  capability, app, server, and video-loader matrix passes 233 tests, two
  optional-runtime skips, and 338 subtests; pinned Ruff E9/F, 66-package
  compatibility, and diff checks pass. Four concurrent fresh app plans
  reported zero remaining bytes. Current free bytes were 364,487,782,400
  against 298,429,691,965 queued reservation bytes and the
  68,719,476,736-byte safety reserve, so the active aggregate queue envelope
  was temporarily 2,661,386,301 bytes over the limit while an existing transfer
  progressed. No Wan bytes were needed; no POST, interruption, or deletion
  occurred, and all cached documentation/example media remains preserved.
- [x] **P6.78 Bound JoyAI Image Edit app download selections:** backend
  `c0c5944` binds both admitted immutable JoyAI repositories to their exact
  safetensors component surfaces. Image Edit retains 38 files /
  50,347,541,462 bytes; Image Edit Plus retains 29 files / 50,338,627,834
  bytes. The allowlists preserve their Qwen3-VL processors/text encoders,
  tokenizers, transformers, Wan VAEs, schedulers, configs, indexes, cards, and
  attribute receipts while excluding seven test images from Edit and three
  example images plus repository `inference.py` from Plus: 16,592,417 bytes
  from the combined 100,702,761,713-byte trees. Both generic adapters already
  explicitly require safe serialization.

  App plan and POST regressions bind both capability-derived selections; a
  focused JoyAI artifact regression requires every indexed component, rejects
  repository code, examples, tests, and legacy serialization surfaces, and
  verifies both existing loader flags. The JoyAI, capability, app, and loader
  matrix passes 220 tests, three optional-runtime skips, and 895 subtests;
  pinned Ruff E9/F, 66-package compatibility, and diff checks pass. Concurrent
  fresh app plans reported Edit complete with zero remaining bytes and Plus
  with 50,315,694,156 selected bytes still outstanding while its pre-existing
  repository-wide app job remained active. Current free bytes were
  363,886,616,576 against 297,828,399,222 queued reservation bytes and the
  68,719,476,736-byte safety reserve, leaving the active aggregate envelope
  2,661,259,382 bytes over the limit. The differently scoped active Plus job
  was not joined, replaced, cancelled, or interrupted; no new POST or deletion
  occurred.
- [ ] Evaluate large image/cascaded families and DiffusionGemma only on hardware
  with sufficient RAM, VRAM, and disk.
  - [x] **DiffusionGemma immutable source/artifact review:** backend `42b609e`
    seals official ungated
    `google/diffusiongemma-26B-A4B-it@f7f5b7f5fa82ffc52addd066915886d497f5517b`.
    The repository contains no Python and requires no remote code. Its exact
    bfloat16 surface is 11 safetensors shards / 51,647,701,024 weight bytes;
    22 files occupy 51,680,024,015 bytes in total. The Apache-2.0 model index
    selects package-owned `DiffusionGemmaForBlockDiffusion`, `Gemma4Processor`,
    `BlockRefinementScheduler`, and `DiffusionGemmaPipeline` classes. Exact
    metadata, weight, pinned Diffusers source, and locked Transformers 5.14.1
    source hashes are recorded in
    `data/diffusiongemma-artifact-review.json`; a no-weight API probe passed.

    The reviewed recipe keeps the 256-token canvas/output, 48 denoising steps
    per canvas, entropy-bound 0.1, temperature decay 0.8 to 0.4, stability 1,
    confidence threshold 0.005, static cache, and compiled decoder. The pinned
    Diffusers pipeline exposes per-step callbacks, but validates only positive
    generation length and step counts; it has no upper bounds for prompt,
    generation, image, or video-frame resources. MoDiff has no generic
    diffusion-text node yet. Accordingly the candidate remains remote-only and
    absent from runtime/download catalogs and every user-facing capability.
    Backend-owned bounds, multimodal input/output safety policy, exact remote
    heavy-hardware measurements, and physical macOS evidence remain pending.
  - [x] **Stable Cascade immutable artifact/source review:** backend `b55983b`
    seals the exact public prior snapshot
    `stabilityai/stable-cascade-prior@7ca32c21c3b4d4e35bbb94fcfedfb4fa2259bd91`
    and decoder snapshot
    `stabilityai/stable-cascade@a89f66d459ae653e3b4d4f992a7c3789d0dc4d16`.
    Neither repository contains Python or requires remote code. The selected
    full-size bf16 partition is six safetensors files / 13,728,020,596 bytes;
    exact component hashes, canonical selected/full inventory digests, model
    indexes, scheduler configuration, pinned Diffusers sources, and the
    identical repository license bytes are recorded in
    `data/stable-cascade-artifact-review.json` without downloading weights.

    This family is not admitted. Its Stability AI Non-Commercial Research
    Community License prohibits the production and hosted-service uses MoDiff
    cannot silently assume. In addition, all three upstream Stable Cascade
    pipelines are deprecated after Diffusers 0.35.2 while MoDiff pins a later
    revision. The combined loader names the connected prior repository but not
    its immutable revision, so it cannot safely identify the two distinct
    snapshots with one shared revision. Any future reviewed path must load the
    prior and decoder independently, pass image embeddings explicitly, and
    retain the reviewed 1024px/20-step prior plus 10-step decoder recipe. No
    runtime/download catalog, workflow, client, Auto, template, Gallery, or
    generated-media surface was added. License resolution, a maintained
    package-owned pipeline, remote heavy-hardware execution, and physical
    macOS evidence remain pending independently.
  - [x] **DeepFloyd IF immutable artifact/source review:** backend `8d45c9f`
    seals the canonical three-stage route at exact revisions:
    `DeepFloyd/IF-I-XL-v1.0@c03d510e9b75bce9f9db5bb85148c1402ad7e694`,
    `DeepFloyd/IF-II-L-v1.0@609476ce702b2d94aff7d1f944dcc54d4f972901`,
    and
    `stabilityai/stable-diffusion-x4-upscaler@572c99286543a273bfd17fac263db5a77be12c4c`.
    All three repositories contain no Python and require no remote code. The
    reviewed 64px, 256px, and 1024px path selects 11 repository-scoped
    safetensors files / 27,326,661,461 bytes, or 26,718,655,226 bytes after
    deduplicating the identical Stage I/II safety-checker and watermarker
    blobs. Full inventory digests, selected file hashes, immutable metadata
    blob identities, all seven pinned package-owned pipeline source hashes,
    stage defaults, prompt-embedding reuse, safety handoff, and estimate-only
    resource bounds are recorded in
    `data/deepfloyd-if-artifact-review.json`; a no-weight API probe passed.

    The family remains unadmitted. Stage I and II require account/contact
    acceptance of the DeepFloyd License, which limits use and distribution to
    noncommercial research and also restricts data produced by the software.
    Anonymous access can inventory immutable blobs but returns HTTP 401 for the
    gated model indexes and component configurations, so their payloads were
    not treated as reviewed. The pinned pipelines provide a legacy per-step
    callback and enforce prompt token/noise or strength constraints, but do not
    own upper bounds for batch, image count, step count, or Stage I/II output
    dimensions. No runtime/download catalog, workflow, client, Auto, template,
    Gallery, or generated-media surface was added. License acceptance,
    authenticated configuration review, backend-owned bounds, remote
    heavy-hardware execution, and physical macOS evidence remain pending.
  - [x] **PixArt Sigma 1024px source admission:** backend `4fd1a66` and
    client `235c9d3` admit the exact public snapshot
    `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS@e102b3591cc82e97071b8b4cb90d834d0c487207`
    through the generic Diffusers image facade. The OpenRAIL++ repository has
    no Python, requires no remote code, and selects four safetensors files /
    21,827,405,446 bytes. Exact file hashes, canonical inventory digest,
    metadata and pinned package-source hashes, the 1024px/20-step/guidance-4.5
    recipe, 300-token bound, and estimate-only resource envelope are sealed in
    `data/pixart-sigma-artifact-review.json` without downloading weights.

    The workflow is Expert-only and remote-only, with Auto and Gallery
    disabled. Its 104-workflow manifest entry is graph-qualified but explicitly
    runtime-unqualified and requires only the immutable PixArt repository.
    Backend `8bca634` and client `60b0269` also keep declarative image contract
    refreshes on the non-installing base path, clear unused auxiliary model
    identities, and finalize the exact loader-issued image contract. The full
    1,373-test backend overlay gate, complete client check, focused base-runtime
    matrix, deterministic workflow verification, and the three affected mocked
    Studio cases pass. Remote real-weight execution, output safety and quality
    review, and physical macOS execution remain pending independently.
  - [x] **AuraFlow v0.3 source admission:** backend `8117b39` and client
    `7d52469` admit the exact public snapshot
    `fal/AuraFlow-v0.3@2cd8588f04c886002be4571697d84654a50e3af3`
    through the generic Diffusers image facade. The Apache-2.0 repository has
    no Python, requires no remote code, and selects the four-file fp16
    safetensors partition / 16,835,036,374 bytes. Exact file hashes, canonical
    inventory digest, immutable metadata and pinned package-source hashes, the
    native 1536x768/50-step/guidance-3.5 recipe, 256-token bound, and
    estimate-only resource envelope are sealed in
    `data/auraflow-v0.3-artifact-review.json` without downloading weights.

    The workflow is Expert-only and remote-only, with Auto and Gallery
    disabled. Its 105-workflow manifest entry is graph-qualified but explicitly
    runtime-unqualified and requires only the immutable AuraFlow repository.
    Backend-owned field contracts and preflight now enforce AuraFlow's 1536px
    maximum side, while client `c5e02ea` updates the restored-queue test fixture
    to model the current loader-issued image-contract signal. The 1,399-test
    exact pinned backend overlay, complete client check, deterministic workflow
    verification, and corrected focused mocked Studio case pass. Remote
    real-weight execution, output safety and quality review, and physical macOS
    execution remain pending independently; no weights or media were
    downloaded or retained.
  - [x] **Bria 3.2 source and admission-gate review:** backend `30a4667`
    records the package-owned `BriaPipeline` and `BriaTransformer2DModel`
    sources at the pinned Diffusers revision, their exact source hashes,
    no-weight call signatures, callback surface, 1024px/30-step/guidance-5
    defaults, 128-token default and 512-token package maximum, and the public
    `briaai/BRIA-3.2` file-tree observations in
    `data/bria-3.2-source-review.json`.

    The family is not admitted. The official repository is gated, labels the
    weights for non-commercial use, links CC BY-NC 4.0, and directs commercial
    users to a separate paid agreement. Anonymous model APIs and file payloads
    return HTTP 401, so an immutable head, exact artifact sizes and hashes,
    model index, component configurations, and repository license payload
    cannot be treated as reviewed. The pinned inference recipe also requires
    bf16 T5 placement with every final `DenseReluDense.wo` projection restored
    to float32 plus a float32 VAE when its shift factor is zero; the generic
    loader does not currently own a validated per-layer dtype contract. No
    runtime/download catalog, workflow, client, Auto, template, Gallery, or
    generated-media surface was added. Authenticated artifact and license
    review, a backend-owned precision/bounds contract, remote heavy-hardware
    execution, and physical macOS evidence remain pending independently. The
    focused 10-test review/catalog matrix and static checks pass; no weights or
    media were downloaded.
  - [x] **Bria FIBO generation/edit source and admission-gate review:** backend
    `a826a5e` seals `briaai/FIBO` at
    `185d92d046a8bc2f93843083894a9ea983b90b32` and
    `briaai/Fibo-Edit` at
    `1bee4ae9b119d2634ec9f378370f2a1ae49d89cc`. The exact current
    safetensors partitions are respectively five files / 25,540,786,720 bytes
    and five files / 24,131,409,512 bytes. The Edit repository's two archived
    transformer shards are kept separate, bringing its complete seven-file
    weight surface to 40,703,183,416 bytes. Canonical inventory digests,
    individual hashes, immutable metadata blob identities, package-owned
    generation/edit/transformer source hashes, structured-JSON call contracts,
    mask path, 3,000-token package bound, callbacks, and current-versus-archive
    identities are recorded in `data/bria-fibo-source-review.json`.

    The family is not admitted. Both model repositories are gated and their
    model cards limit the public weights to non-commercial use while directing
    commercial users to a separate agreement; authenticated configs and exact
    gated license payloads return HTTP 401. The official quality path also uses
    two CC-BY-NC-4.0 custom-code Modular promptifiers. Their exact code revisions
    and hashes are sealed, but they require `trust_remote_code=True`, hard-code
    `.to("cuda")`, expose an unbounded 4,096-token sampling default, and load
    the nested 8,875,719,328-byte FIBO VLM snapshots without passing their
    immutable revisions. Generation accepts arbitrary strings despite requiring
    structured JSON for reviewed quality; Edit validates an `edit_instruction`
    JSON but the local promptifier does not cover the documented mask route.
    No runtime/download catalog, workflow, client, Auto, template, Gallery, or
    generated-media surface was added. License acceptance, authenticated config
    review, explicit remote-code authorization, backend-owned JSON/device/model
    pinning and bounds, remote heavy-hardware execution, and physical macOS
    evidence remain pending independently. The focused review matrix and static
    checks pass; no weights or media were downloaded.
  - [x] **Chroma1-HD text-to-image source admission:** backend `5d3bc8f` and
    client `86cbd92` admit the exact public snapshot
    `lodestones/Chroma1-HD@0e0c60ece1e82b17cb7f77342d765ba5024c40c0`
    through the generic Diffusers image facade. The model-card metadata declares
    Apache-2.0, the repository has no Python and requires no remote code, and the
    selected Diffusers partition contains five safetensors files /
    27,492,403,238 bytes. The duplicate 17,800,038,288-byte original single-file
    distribution is excluded. Exact selected and excluded file hashes, canonical
    inventory digest, immutable metadata and shard-index identities, and pinned
    package-owned text-to-image, image-to-image, output, and transformer source
    hashes are sealed in `data/chroma1-hd-artifact-review.json` without
    downloading weights.

    The admitted workflow is Expert-only and remote-only. Backend-owned
    contracts bound it to the reviewed bfloat16 1024x1024/40-step/guidance-3
    recipe and at most 512 prompt tokens. Its 106-workflow manifest entry is
    graph-qualified but explicitly runtime-unqualified, while Chroma
    image-to-image, Auto, and Gallery remain disabled. The upstream model card
    explicitly states that the model has no safety alignment, so live output
    safety and quality review remains a hard gate. The complete 1,413-test
    backend overlay with 3,110 subtests, project static checks, complete client
    check, and deterministic workflow verification pass. Remote real-weight
    execution and physical macOS execution remain pending independently; no
    weights or media were downloaded or retained.
  - [x] **CogView3 Plus 3B text-to-image source admission:** backend `f74c806`
    and client `a4d0b99` admit the exact public snapshot
    `zai-org/CogView3-Plus-3B@5d70e40732ac0efac98524c51a7fa9c82707f1e5`
    through the generic Diffusers image facade. The repository is package-owned,
    contains no Python, requires no remote code, and exposes seven bfloat16
    safetensors files / 25,559,227,422 bytes. Exact file hashes, canonical
    inventory digest, immutable metadata identities, 2,848,836,672-parameter
    safetensors metadata, and pinned package pipeline/output/transformer source
    hashes are sealed in `data/cogview3-plus-3b-artifact-review.json` without
    downloading weights. The model-card metadata declares Apache-2.0 and links
    `LICENSE.md`, but neither that path nor `LICENSE` exists in the immutable
    tree; this missing linked license file is recorded rather than silently
    treated as stronger repository evidence.

    The admitted text-to-image workflow is Expert-only and remote-only.
    Backend-owned contracts require bfloat16, constrain both sides to
    512-2048 pixels in 32-pixel increments, cap inference at 50 steps and 224
    prompt tokens, and preserve the reviewed 1024x1024/guidance-7 recipe. The
    upstream A100 memory figures are recorded as estimates only. The
    deterministic 107-workflow catalog is graph-qualified but explicitly
    runtime-unqualified; Auto and Gallery remain disabled. The complete
    1,419-test backend overlay with 3,130 subtests and three platform skips,
    project static checks, complete client check, and deterministic workflow
    verification pass. Remote real-weight output safety/quality review and
    physical macOS execution remain pending independently; no weights or media
    were downloaded or retained.
  - [x] **CogView4 6B text-to-image source admission:** backend `495d07d` and
    client `9822baa` admit the exact public snapshot
    `zai-org/CogView4-6B@63a52b7f6dace7033380cd6da14d0915eab3e6b5`
    through the generic Diffusers image facade. The immutable repository carries
    the complete Apache License 2.0 text, contains no Python, requires no remote
    code, and exposes eight bfloat16 safetensors files / 31,108,954,670 bytes.
    Exact file hashes, canonical inventory digest, immutable metadata identities,
    6,369,118,272-parameter safetensors metadata, and pinned package
    pipeline/output/transformer source hashes are sealed in
    `data/cogview4-6b-artifact-review.json` without downloading weights.

    The admitted text-to-image workflow is Expert-only and remote-only.
    Backend-owned contracts require bfloat16, constrain both sides to
    512-2048 pixels in 32-pixel increments, enforce the model card's 2^21-pixel
    ceiling across both dimensions, and cap inference at 50 steps and 1,024
    prompt tokens while preserving the reviewed 1024x1024/guidance-3.5 recipe.
    The package signature's 1,024-token default disagrees with its 224-token
    docstring, and the model card's 1920x1280 memory row exceeds its own stated
    2^21-pixel ceiling. Both contradictions are recorded; the executable route
    follows the actual signature and the stricter declared pixel ceiling. The
    upstream A100 batch-four memory figures remain estimates only. The
    deterministic 108-workflow catalog is graph-qualified but explicitly
    runtime-unqualified; Auto and Gallery remain disabled. The complete
    1,425-test backend overlay with 3,150 subtests and three platform skips,
    dependency/preflight/static checks, complete client check, and deterministic
    workflow verification pass. Remote real-weight output safety/quality review
    and physical macOS execution remain pending independently; no weights or
    media were downloaded or retained.
  - [x] **VisualCloze source and admission-gate review:** backend `0f4d3c4`
    seals the exact public full-model snapshots
    `VisualCloze/VisualClozePipeline-384@59c469d2772d927ffe55f3543c4d3bd556fd46a4`
    and
    `VisualCloze/VisualClozePipeline-512@feaad2dd83d3d42bad197b9d31fe2f6c5b4cb1bb`.
    Each immutable repository contains seven bfloat16 safetensors files /
    33,743,379,958 bytes, no Python, and no remote-code requirement. Exact file
    hashes, canonical inventory digests, immutable metadata identities,
    11,902,391,360-parameter safetensors metadata, and pinned package-owned
    combined/generation/processor source hashes are recorded in
    `data/visualcloze-artifact-review.json` without downloading weights. Both
    model cards declare Apache-2.0 in metadata, but neither immutable tree
    contains a license file. The two 2,482,363,148-byte legacy `.pth` LoRA
    checkpoints are separately sealed and excluded under the safe-serialization
    policy; the full Diffusers snapshots do not require them.

    This family is deliberately review-only. Its task input is a nested,
    rectangular image matrix containing one or more explicit null target cells,
    with distinct task/content prompts and an optional second SDEdit upsampling
    stage. Mapping that shape onto an existing single-image edit or inpaint
    alias would be incorrect. The pinned package supplies no maximum batch,
    row, column, cumulative input-pixel, denoising-step, or upsampling-dimension
    bounds. Accordingly no runtime/download catalog, workflow, client, Auto,
    template, or Gallery surface was added. A generic visual-context-matrix
    contract, backend-owned resource bounds, license-file clarification, remote
    heavy-hardware execution, live output review, and physical macOS evidence
    remain independent gates. The complete 1,430-test backend overlay with
    3,154 subtests and three platform skips, project static/dependency/preflight
    checks, and the focused immutable review matrix pass; no weights or media
    were downloaded or retained.
  - [x] **Allegro text-to-video source admission:** backend `6488462` and
    client `05a2e15` admit the exact public snapshot
    `rhymes-ai/Allegro@c1b9207bb5cb79e2aa08f3d139c17d26c0de55b6`
    through the generic Diffusers video facade. The immutable repository
    contains no Python and requires no remote code. Six selected bfloat16
    safetensors files / 25,293,069,108 bytes, their exact hashes and canonical
    inventory digest, immutable metadata identities, 2,771,907,856-parameter
    safetensors metadata, and pinned package pipeline/output/transformer/VAE
    source hashes are sealed in `data/allegro-artifact-review.json` without
    downloading weights. The duplicate 19,049,317,384-byte unsafe PyTorch `.bin`
    text-encoder partition is explicitly excluded. The model card declares
    Apache-2.0, but the immutable repository contains no license file.

    The admitted text-to-video workflow is Expert-only and remote-only. Its
    backend-owned contract preserves the reviewed native 1280x720, 88-frame,
    100-step, guidance-7.5, 512-token, 15-FPS recipe; requires bfloat16 for the
    text encoder and transformer; keeps the VAE in float32 with mandatory
    tiling; and defaults to sequential CPU offload. Conservative planning
    reserves 10 GiB accelerator memory, 30 GiB disk, and 48 GiB system RAM.
    The deterministic 109-workflow catalog is graph-qualified but explicitly
    runtime-unqualified; Auto and Gallery remain disabled. The complete
    1,441-test backend overlay with 3,175 subtests and three platform skips,
    dependency/preflight/static checks, complete client check, deterministic
    workflow verification, and the nine-byte remaining client bundle margin
    pass. Remote real-weight output safety/quality review and physical macOS
    execution remain pending independently; no weights or media were downloaded
    or retained.
  - [x] **AnyFlow source and admission-gate review:** backend `09d3c98` seals
    all four official public Diffusers snapshots: bidirectional Wan2.1 T2V 1.3B
    at `4c2ec05c7fa4dbafbca131ad32430905c7ff2974`, bidirectional T2V 14B at
    `ed91e001c08a88df8bbdc18f29b43b8078459627`, FAR 1.3B at
    `915af337434035df8545797ecc910d79fa78cf29`, and FAR 14B at
    `6207c4512a306d2a5a564df66b04a78668923740`. The exact seven-file /
    26,074,853,036-byte and 26,075,642,740-byte 1.3B inventories and nine-file /
    51,863,430,540-byte and 51,866,062,428-byte 14B inventories are bfloat16
    safetensors only. All repositories contain complete identical license files,
    no Python, and no remote-code requirement. Immutable metadata, model-card,
    model-index, scheduler, transformer, VAE, artifact, and pinned package source
    hashes are recorded in `data/anyflow-artifact-review.json` without
    downloading weights.

    This family is deliberately review-only. The NVIDIA One-Way Noncommercial
    License restricts the models and derivatives to non-commercial research
    activities or publications, so legal product-admission approval remains a
    hard gate. The package-owned bidirectional T2V and FAR T2V/I2V/V2V contracts,
    canonical 832x480/81-frame/4-step/guidance-1/16-FPS recipe, FAR chunk
    partition, callback surface, and missing backend resource bounds are sealed.
    The immutable model cards still import custom upstream pipelines and use
    `context_sequence`, whereas the current package-owned API uses `video`; that
    mismatch is explicit. No runtime/download catalog, capability, workflow,
    client, Auto, template, or Gallery surface was added. The complete
    1,446-test backend overlay with 3,183 subtests and three platform skips plus
    static/dependency/preflight checks pass. Legal approval, bounded contracts,
    remote execution, live output review, and physical macOS evidence remain
    independent gates; no weights or media were downloaded or retained.
  - [x] **ChronoEdit source and admission-gate review:** backend `70640ae`
    seals the exact public snapshot
    `nvidia/ChronoEdit-14B-Diffusers@26b33e0d056203dc30c733ad02b86ac225eb17d5`.
    Its required package-owned core is 21 safetensors files /
    90,075,130,404 bytes; exact file hashes, canonical inventory digest,
    immutable model-card/component/index identities, 16,394,878,784-parameter
    Hub metadata, and pinned pipeline/output/transformer source hashes are
    recorded in `data/chronoedit-artifact-review.json`. The included 8-step
    distillation LoRA and the two exact public upscaler/paint-brush LoRA
    revisions are separately sealed but not selected. The model repository is
    public and contains no Python or remote-code requirement, but it has no
    embedded license file.

    This family is deliberately review-only. The immutable card names the
    mutable external NVIDIA Open Model License Agreement as governing terms and
    says rights terminate if a contained safety guardrail is bypassed, disabled,
    circumvented, or made less effective. The package-owned Diffusers pipeline
    has no safety checker, while the bundled 102-file / 7,171,449,905-byte Cosmos
    guardrail includes exact `.pth` and `.pt` artifacts that fail MoDiff's safe
    serialization policy. The artifact model index also still names the generic
    Wan pipeline/transformer and requires explicit ChronoEdit component
    overrides. Native five-frame editing, 29-frame temporal reasoning, and
    eight-step distilled recipes are sealed, along with the final-frame image
    semantics and missing backend bounds. No runtime/download catalog,
    capability, workflow, client, Auto, template, or Gallery surface was added.
    The complete 1,453-test backend overlay with 3,183 subtests and three
    platform skips plus static/dependency/preflight checks pass. Immutable or
    approved governing terms, a complete safe guardrail integration, a generic
    edit-plus-reasoning contract, remote execution, live output review, and
    physical macOS evidence remain independent gates; no weights or media were
    downloaded or retained.
  - [x] **ConsisID source and admission-gate review:** backend `a52c7ec`
    seals the exact public snapshot
    `BestWishYsh/ConsisID-preview@950bc3f0902db44799e223a12ad972f9c52b341d`.
    Its generator contains five bfloat16 safetensors files /
    22,821,396,692 bytes, no Python, and no remote-code requirement. Exact file
    hashes, canonical inventory digest, immutable model-card/component/index
    identities, 6,217,102,912-parameter Hub metadata, and pinned package
    pipeline/output/transformer/face-utility source hashes are recorded in
    `data/consisid-artifact-review.json`. The official Diffusers documentation
    also names `BestWishYsh/ConsisID-1.5`, but that repository returns not found
    to unauthenticated metadata lookup as of 2026-08-13 and is not invented as
    an available artifact. The preview card declares Apache-2.0 in metadata but
    the immutable tree contains no license file.

    This family is deliberately review-only. Its required package-owned face
    preparation hard-requires InsightFace, FaceXLib, a custom EVA-CLIP package,
    OpenCV, TorchVision, and ONNX Runtime; hardcodes CUDA execution providers;
    and consumes an eight-artifact / 1,446,798,634-byte identity stack containing
    five ONNX models plus three required unsafe `.pt`/`.pth` files. Four more
    unused unsafe preprocessing/face artifacts are separately sealed. The
    package describes identity tensors as crucial but does not require them in
    pipeline input validation, so exposing the generator alone would silently
    defeat the family's identity-preserving contract. Biometric privacy,
    consent, and misuse controls are also unresolved. No runtime/download
    catalog, capability, workflow, client, Auto, template, or Gallery surface
    was added. The complete 1,460-test backend overlay with 3,183 subtests and
    three platform skips plus static/dependency/preflight checks pass. A safe
    cross-platform face stack, bounded generic identity-video contract, license
    clarification, remote execution, live identity/safety review, and physical
    macOS evidence remain independent gates; no weights or media were downloaded
    or retained.
  - [x] **Latte text-to-video source admission:** backend `180917e` and client
    `847ed6c` admit the exact public snapshot
    `maxin-cn/Latte-1@0653024365272f061fc44d1078134df22842b687`
    through the generic Diffusers video facade. The immutable repository
    contains no Python and requires no remote code. Six selected safetensors
    files / 23,614,979,636 bytes, their exact hashes and canonical inventory
    digest, immutable component/model-card/model-index identities, Hub
    safetensors metadata, and pinned package pipeline/transformer/VAE source
    hashes are sealed in `data/latte-artifact-review.json` without downloading
    weights. The 4,231,339,889-byte unsafe legacy `.pt` checkpoint and the
    391,017,740-byte optional temporal VAE that is not referenced by the native
    model index are explicitly excluded. The immutable card declares
    Apache-2.0, but the repository contains no license file.

    The admitted text-to-video workflow is Expert-only and remote-only. Its
    backend-owned contract preserves the native 512x512, 16-frame, 50-step,
    guidance-7.5, 120-token, 8-FPS recipe; requires float16 safe loading; uses
    one output, raw-caption encoding, the documented feature mask and temporal
    attentions, 14-frame decode chunks, callbacks, and sequential CPU offload;
    and rejects every image, video, and mask input. The package signature's
    50-step/guidance-7.5 defaults are authoritative and the conflicting
    100-step/guidance-7.0 docstring values are recorded explicitly.
    Conservative estimate-only planning reserves 16 GiB accelerator memory,
    32 GiB disk, and 48 GiB system RAM because upstream reports A100 timing but
    no peak-memory measurement. The deterministic 110-workflow catalog is
    graph-qualified but explicitly runtime-unqualified; Auto and Gallery remain
    disabled. The complete 1,471-test backend overlay with 3,204 subtests and
    three platform skips, Ruff E9/F, compile, 66-package compatibility,
    preflight, complete client check, and deterministic workflow verification
    pass. The unchanged client bundle ceiling passes at 530,428 / 530,432 gzip
    bytes. Remote real-weight output safety/quality review, license-file
    clarification, and physical macOS execution remain pending independently;
    no weights or media were downloaded or retained.
  - [x] **Lucy Edit source and admission-gate review:** backend `6ab1033`
    seals the exact public snapshot
    `decart-ai/Lucy-Edit-Dev@cb201fdec1bca6e7c362e127392c1632c92d2576`.
    Its package-owned Diffusers core contains five float32 safetensors files /
    34,182,223,896 bytes, no Python, and no remote-code requirement. Exact file
    hashes, canonical inventory digest, immutable model-card/component/index
    identities, 5,000,377,536-parameter Hub metadata, the externally linked
    license PDF hash retrieved on 2026-08-13, and pinned package
    pipeline/output/Wan-transformer/Wan-VAE source hashes are recorded in
    `data/lucy-artifact-review.json`. The model repository is public and
    ungated, but contains no license file; its governing license and incorporated
    acceptable-use policy are mutable external documents.

    This family is deliberately review-only. The Lucy Edit 5B Model Community
    License permits only non-commercial, non-production use, excludes commercial
    use of outputs, and defines hosted remote access as distribution. That is
    incompatible with admission to MoDiff's product runtime without a separate
    commercial license and explicit legal approval. The native 832x480,
    81-frame, 50-step, guidance-5, 512-token, 24-FPS edit recipe, callback and
    sequential-offload surfaces, BF16-pipeline/FP32-VAE load recommendation, and
    missing backend resource bounds are sealed. The package also normalizes
    `num_frames` but never uses it to select or validate the input video's frame
    count. No runtime/download catalog, capability, workflow, client, Auto,
    template, or Gallery surface was added. The complete 1,476-test backend
    overlay with 3,204 subtests and three platform skips plus
    static/dependency/preflight checks pass. Legal product-admission approval,
    immutable governing terms, bounded execution, remote qualification, live
    output review, and physical macOS evidence remain independent gates; no
    weights or media were downloaded or retained.
  - [x] **Mochi 1 Preview text-to-video source admission:** backend `fb3e39f`
    and client `c0afea5` admit the exact public snapshot
    `genmo/mochi-1-preview@14be5fcea23095ed330cb214647916a451e38b6e`
    through the generic Diffusers video facade. The immutable repository
    contains no Python and requires no remote code. Eight selected safetensors
    files / 40,024,303,350 bytes, their exact hashes and canonical inventory
    digest, immutable model-card/component/index identities, 10,027,677,744-
    parameter Hub metadata, and pinned package pipeline/output/transformer/VAE
    source hashes are sealed in `data/mochi-artifact-review.json` without
    downloading weights. Duplicate original-format weights, an unindexed
    two-shard T5 copy, and the default float32 transformer and VAE partitions
    are explicitly excluded. The immutable card declares Apache-2.0, but the
    repository contains no license file.

    The admitted text-to-video workflow is Expert-only and remote-only. Its
    backend-owned contract preserves the official native 848x480, 31-frame,
    64-step, guidance-4.5, 256-token, 30-FPS recipe; requires the BF16 variant,
    explicitly preloads the indexed T5 encoder, mandates VAE tiling, uses one
    output with callbacks and sequential CPU offload, and rejects every image,
    video, and mask input. The repository card's conflicting 84-frame example,
    the package signature's 19-frame default, and its internally inconsistent
    step documentation are recorded explicitly; exact 31-frame temporal-VAE
    congruence is retained. Conservative estimate-only planning preserves the
    publisher's 22-60 GiB accelerator-memory range. The deterministic
    111-workflow catalog is graph-qualified but explicitly runtime-unqualified;
    Auto and Gallery remain disabled. The complete 1,486-test backend overlay
    with 3,269 subtests and three platform skips, Ruff E9/F, compile,
    66-package compatibility, preflight, complete client check, and
    deterministic workflow verification pass. The unchanged client bundle
    ceiling passes at 530,428 / 530,432 gzip bytes. Remote real-weight output
    safety/quality review, license-file clarification, and physical macOS
    execution remain pending independently; no weights or media were downloaded
    or retained.
  - [x] **SANA-Video 2B 480p text/image-to-video source admission:** backend
    `081a083` and client `6ed67bf` admit the exact public snapshot
    `Efficient-Large-Model/SANA-Video_2B_480p_diffusers@db5f398b13ca086d09a50ce156c20527773841b1`
    through the generic Diffusers video facade. The immutable repository is
    ungated, contains a complete Apache-2.0 license file, contains no Python,
    and requires no remote code. Its exact five-file / 13,963,813,420-byte
    mixed BF16/FP32 safetensors partition, canonical inventory digest,
    immutable metadata identities, parameter counts from safetensors headers,
    and pinned package pipeline/output/transformer/Wan-VAE/scheduler source
    hashes are sealed in `data/sana-video-artifact-review.json` without
    downloading weights. The unsafe original 480p and 720p `.pth` repositories,
    the distinct 18.35 GB safe 720p partition, and the duplicate-heavy separate
    LongLiveSANA contract are explicitly excluded.

    The two admitted workflows are Expert-only and remote-only. Their
    backend-owned contracts preserve the official native 832x480, 81-frame,
    50-step, guidance-6, 300-token, 16-FPS recipe; append the reviewed motion
    score of 30; keep the transformer and text encoder in BF16; preload the Wan
    VAE in FP32; mandate VAE tiling and sequential CPU offload; disable
    resolution binning; and distinguish text-only input from exactly one I2V
    opening image. The pinned package's MPS rotary-frequency float32 workaround
    is recorded, as is its decode-OOM branch that may not produce a decoded
    value; mandatory tiling does not overstate live qualification. Conservative
    estimate-only planning reserves 24 GiB accelerator memory, 24 GiB selective
    disk, and 48 GiB system RAM. The deterministic 113-workflow catalog is
    graph-qualified but explicitly runtime-unqualified; Auto and Gallery remain
    disabled. The client stays within its unchanged bundle ceiling by sharing
    equivalent offload constants and removing only prose duplicated by typed
    media requirements. The complete 1,498-test backend overlay with 3,270
    subtests and three platform skips, Ruff E9/F, compile, 66-package
    compatibility, preflight, complete client check, and deterministic workflow
    verification pass at 530,406 / 530,432 gzip bytes. Remote real-weight output
    safety/quality review and physical macOS execution remain pending
    independently; no weights or media were downloaded or retained.
  - [x] **Hunyuan-DiT v1.2 ControlNet Canny source admission:** backend
    `72185d0` and client `f134f98` admit the exact public distilled base
    `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled@ba991d1546d8c50936c4c16398ed0a87b9b99fb1`
    with the exact Canny component
    `Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Canny@b2d21391ebcf78939344cfec84891932f9d53aa0`
    through the generic Diffusers image facade. Both snapshots are ungated,
    safetensors-only for the admitted partition, contain no Python, and require
    no remote code. Their exact six-file / 17,399,623,404-byte float32 weight
    inventory, canonical digest, safetensors-header parameter counts, immutable
    metadata identities, and pinned package pipeline/model/VAE/scheduler/output
    source hashes are sealed in
    `data/hunyuan-dit-controlnet-artifact-review.json` without downloading
    weights. The exact Depth and Pose component revisions are cataloged as
    reviewed Expert substitutions but intentionally have no canonical graph.

    The shared Tencent Hunyuan Community License is sealed from its immutable
    14,614-byte source and requires one explicit client acknowledgement for the
    base-plus-ControlNet assembly. Its 100-million-MAU commercial threshold,
    distribution notices, public machine-generation disclosure, output-use,
    acceptable-use, and military-use restrictions are recorded without
    overstating rights. The admitted Expert-only, remote-only graph preserves
    the official 1024x1024, 50-step, guidance-6, scale-1 Canny recipe, both text
    ceilings, float16 loading, exact artifact revisions, model CPU offload,
    and bounded control preprocessing. The base declares a required safety
    checker but ships none, so Auto and Gallery remain disabled. The
    deterministic 114-workflow catalog is graph-qualified but explicitly
    runtime-unqualified. The complete 1,505-test backend overlay with 3,289
    subtests and three platform skips, Ruff E9/F, compile, 66-package
    compatibility, modular-contract check, preflight, complete client check,
    and deterministic workflow verification pass at 530,393 / 530,432 gzip
    bytes. Remote real-weight memory/output safety/quality review and physical
    macOS execution remain pending independently; no weights or media were
    downloaded or retained.
  - [x] **Stable Diffusion 3 ControlNet source and admission-gate review:** the
    pinned package contains official package-owned Canny/Tile and inpainting
    pipelines, `SD3ControlNetModel`, flow-matching scheduler, callback support,
    model CPU offload, and finite prompt/resolution/control bounds. The exact
    gated base
    `stabilityai/stable-diffusion-3-medium-diffusers@ea42f8cef0f178587cf766dc8129abd379c90671`,
    public InstantX Canny and Tile revisions, and public Alimama inpainting
    revision are sealed in `data/sd3-controlnet-artifact-review.json`. The
    review records the base's six-file / 15,499,002,486-byte selected fp16
    safetensors partition, all three auxiliary weight hashes and header-derived
    parameter counts, three reproducible assembly digests, metadata and package
    source hashes, native 1024px/28-step recipes, and estimate-only resource
    envelopes. Only 125,040 safetensors-header bytes were fetched; no full
    weights or media were downloaded.

    Admission is deliberately blocked. The required base's model index and
    component configs return HTTP 401 without accepted authenticated access,
    so the executable partition cannot be fully reviewed here. Its exact
    immutable license is non-commercial-only and forbids production and
    hosted/API use without a separate license. The Canny and Tile repositories
    publish neither license metadata nor a license file. The inpainting
    repository's copied Stability Community License notice does not
    unambiguously reconcile its derivative-weight grant with the exact older
    base license. The package pipelines also have no safety checker or
    equivalent output guardrail. Accordingly no runtime/download catalog,
    capability, canonical graph, client, Auto, template, or Gallery surface is
    added. Authenticated artifact review, auxiliary rights resolution, legal
    product approval, backend-owned optional-runtime and execution bounds,
    remote heavy-hardware safety/quality review, and physical macOS execution
    remain independent gates.
  - [x] **DiT source and admission-gate review:** the pinned package contains
    the official package-owned `DiTPipeline`, `DiTTransformer2DModel`, DDIM
    scheduler, fixed ImageNet class-label contract, model CPU offload, and
    finite 256px and 512px shapes. The only two exact Facebook repositories,
    `facebook/DiT-XL-2-256@eab87f77abd5aef071a632f08807fbaab0b704d0`
    and
    `facebook/DiT-XL-2-512@101a3d462b22d64c4afdd4d0c8c59a2c0b961b99`,
    are sealed in `data/dit-artifact-review.json`. The review records both
    8-file / 3,334,284,997-byte repositories, all four immutable weight
    identities, repository metadata and package source hashes, exact
    1,000-class input/output contracts, the package's 25/50/250-step example,
    call-default, and docstring discrepancy, and estimate-only resource
    envelopes. No weight bytes or media were downloaded.

    Admission is deliberately blocked. Both official snapshots publish their
    3,334,245,438-byte selected partitions only as legacy pickle-based PyTorch
    `.bin` files and publish no safetensors alternative. Both declare CC BY-NC
    4.0 only in model-card metadata, do not bundle a license file, and cannot be
    admitted for commercial product use. The package pipeline has neither a
    safety checker nor a step callback for cooperative cancellation.
    Accordingly no community conversion, runtime/download catalog, capability,
    canonical graph, client, Auto, template, or Gallery surface is added. An
    official safe-serialization snapshot, commercial product rights, a bundled
    license receipt, backend-owned loading/bounds and cancellation, remote
    heavy-hardware safety/quality review, and physical macOS execution remain
    independent gates.
  - [x] **ERNIE Image Turbo text-to-image source admission:** backend
    `a0b07c8` and client `0527a66` admit the exact public Apache-2.0 snapshot
    `baidu/ERNIE-Image-Turbo@bc68c81e2a1730a394d5fc9fae70713dee940140`
    through the generic Diffusers image facade. The immutable repository is
    ungated, bundles its complete license, contains no Python, and requires no
    remote code. Its exact five-file / 31,596,733,630-byte predominantly BF16
    safetensors partition, canonical inventory digest, immutable metadata
    identities, parameter counts from bounded remote headers, and pinned
    package pipeline/output/transformer/VAE/scheduler and Transformers model
    source hashes are sealed in
    `data/ernie-image-turbo-artifact-review.json` without downloading weights.
    The distinct 50-step `baidu/ERNIE-Image` sibling was reviewed but is not
    conflated with or exposed by the Turbo contract.

    The admitted workflow is Expert-only and remote-only. Its backend-owned
    contract fixes the reviewed Turbo path to 1024x1024, 8 steps, guidance 1,
    prompt enhancement enabled, the tokenizer's 2,048-token ceiling, BF16,
    and model CPU offload, while retaining the package step callback for
    cancellation. The package has no safety checker, so Auto and Gallery
    remain disabled. Admission also extends the exact optional Transformers
    profile with `Mistral3Model` and `Ministral3ForCausalLM`; the same audit
    removed `StableAudioPipeline` from the LoRA adapter-method qualification
    set because that class does not implement the declared methods. The
    resulting clean-base Linux profile digest
    `sha256:6ce66835ba72c609ce83b41ddfa3141e95b42118f86cf419f25e2dfb35409809`
    passed locked install validation, activation, the finite CLIP+PEFT child,
    rollback, and clean-base restoration without retaining managed state.
    The deterministic 115-workflow catalog is graph-qualified but ERNIE model
    execution remains unqualified. The complete 1,523-test backend overlay with
    3,315 subtests and three platform skips, Ruff E9/F, complete client check,
    unchanged 530,432-byte gzip ceiling, and deterministic workflow
    verification pass. Remote real-weight memory/output safety/quality review
    and physical macOS execution remain pending independently; no weights or
    media were downloaded or retained.
  - [x] **GLM-Image text-to-image source admission:** backend `c338824` and
    client `d42a15c` admit the exact public snapshot
    `zai-org/GLM-Image@2c433cc0cbc293bde2ac8ca9624f279b5d23fcf4`
    through the generic Diffusers image facade. The immutable repository is
    ungated, contains no Python, requires no remote code, and declares MIT
    terms in its model card; the incorporated `X-Omni` tokenizer weights
    remain Apache-2.0. The missing bundled license/notice file is retained as
    an explicit redistribution clarification rather than silently inferred
    away. Its exact nine-file / 35,765,307,854-byte mixed BF16/FP32
    safetensors partition, canonical inventory digest, immutable metadata
    identities, parameter counts from bounded remote headers, and pinned
    Diffusers/Transformers source hashes are sealed in
    `data/glm-image-artifact-review.json` without downloading weights.

    The admitted workflow is Expert-only and remote-only. Its backend-owned
    contract deliberately admits text-to-image first, fixes the recipe to
    1024x1024, 50 steps, guidance 1.5, a 2,048-token ceiling, BF16 with the T5
    encoder restored to FP32, and model CPU offload, while retaining the
    package step callback for cancellation. Image-to-image and multi-image
    inputs remain outside this initial route. The package exposes neither a
    negative-prompt parameter nor a safety checker, so negative prompt is
    hidden and Auto and Gallery remain disabled. Admission extends the exact
    optional runtime with the package-owned GLM pipeline/transformer and real
    Transformers GLM processor/model/tokenizer symbols; it also closes the
    ERNIE pipeline/transformer/VAE symbol surface missed by P6.33. The revised
    clean-base Linux profile digest
    `sha256:e1f6cc3420630a6591bd623553f6f43bfe84fbaa041d2ef5f3d4812542278ad8`
    passed locked install validation, activation, the finite CLIP+PEFT child,
    rollback, and clean-base restoration without retaining managed state.
    The deterministic 116-workflow catalog is graph-qualified but GLM model
    execution remains unqualified. The complete 1,529-test backend overlay
    with 3,336 subtests and three platform skips, Ruff E9/F, complete client
    check, unchanged 530,432-byte gzip ceiling, and deterministic workflow
    verification pass. Remote real-weight memory/output safety/quality review,
    bundled-license clarification, and physical macOS execution remain pending
    independently; no weights or media were downloaded or retained.
  - [x] **HiDream-I1 source and admission-gate review:** backend `0685ee5`
    seals the exact public Full, Dev, and Fast snapshots at
    `8ccbbfb270ccdae26d6bb0081df67dc81e4033bf`,
    `0fad2ea0ccf9a80ddf019ea777eedb27c1ccb232`, and
    `4856a5d8cd6fbd194780ed9f289bdf696d3afc10`. Each snapshot contains
    an exact 12-file, approximately 47.18 GB safetensors partition with five
    shared CLIP/T5/VAE files and seven variant-specific transformer shards.
    Their immutable weight identities, repository metadata and package source
    hashes, official 50/28/16-step recipes, seven finite resolution presets,
    128-token prompt default, negative-prompt support, model-offload sequence,
    callback/interrupt surface, and estimate-only composite resource envelope
    are sealed in `data/hidream-image-artifact-review.json`. No weight bytes
    or media were downloaded.

    Admission is deliberately blocked rather than treating those safe public
    partitions as runnable by themselves. Every model index requires
    `text_encoder_4` and `tokenizer_4`, but all three immutable trees omit
    those subfolders. The official loader supplies them from
    `meta-llama/Llama-3.1-8B-Instruct@0e9e39f249a16976918f6564b8830bc894c89659`,
    whose manual gate masks all four safetensors identities and returns HTTP
    401 for configuration before authenticated Llama 3.1 license acceptance.
    None of the three model snapshots bundles a composite license/notice file,
    the generic loader cannot yet assemble and receipt that separately pinned
    external tokenizer/causal language model, and the package has no safety
    checker. Accordingly no runtime/download catalog, capability, canonical
    graph, client, Auto, template, or Gallery surface is added. Fresh
    task-scoped Llama terms acceptance, authenticated exact artifact review,
    a product-owned composite license receipt, backend-owned external encoder
    assembly and bounds, remote heavy-hardware safety/quality review, and
    physical macOS execution remain independent gates.
  - [x] **HunyuanImage 2.1 source and territory-gate review:** backend `4987495`
    seals the exact public package-owned conversion
    `hunyuanvideo-community/HunyuanImage-2.1-Diffusers@7e7b7a177de58591aeaffca0929f4765003d7ced`
    and the governing upstream receipt
    `tencent/HunyuanImage-2.1@e435da11d9e8795a25e224c5ba27b099ed45c55b`.
    The conversion contains an exact ten-file / 53,124,614,990-byte BF16
    safetensors partition, no Python, and no remote-code requirement. Its
    immutable weight identities, repository/config hashes, package pipeline,
    refiner, transformer, VAE, and guider source hashes, 2K/50-step/APG-3.5
    first-stage recipe, prompt limits, cancellation surface, and estimate-only
    resource envelope are sealed in
    `data/hunyuan-image-artifact-review.json`. No weight bytes or media were
    downloaded.

    The family remains contract-only. The immutable Tencent license expressly
    excludes the European Union, United Kingdom, and South Korea and prohibits
    using the works or outputs outside that Territory. MoDiff has no
    legal/product-approved territory enforcement spanning download, local and
    hosted execution, output handling, or redistribution. The safe community
    conversion also omits the governing LICENSE and NOTICE and links mutable
    terms, while the package exposes no safety checker. Accordingly no
    runtime/download catalog, capability, graph, client, Auto, template, or
    Gallery surface is added. Legal territory/distribution approval, product
    territory enforcement, an immutable composite terms receipt, remote
    heavy-hardware safety/quality/cancellation qualification, and physical
    macOS execution remain independent gates.
  - [x] **Hunyuan-DiT v1.2 Distilled standalone source admission:** backend
    `1e97362` and client `e6e306f` expose the already reviewed exact public
    snapshot
    `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled@ba991d1546d8c50936c4c16398ed0a87b9b99fb1`
    as a distinct package-owned text-to-image workflow. The same base remains
    independently reusable by the Canny ControlNet assembly; the standalone
    route does not load or silently require that auxiliary component. Its
    exact five-file / 14,422,655,700-byte float32 safetensors inventory,
    canonical digest, immutable metadata/package-source hashes, and sealed
    Tencent terms receipt are recorded in
    `data/hunyuan-dit-artifact-review.json` without downloading weights.

    The backend-owned Expert contract fixes the distilled path to 1024x1024,
    25 steps, package-default guidance 5, BERT/T5 limits of 77/256, float16
    loading, and explicit CPU offload, while retaining the package callback
    and interrupt surface. The snapshot declares a required safety checker
    but ships none, so Auto and Gallery remain disabled. The optional runtime
    now explicitly validates the base and ControlNet HunyuanDiT pipeline and
    transformer symbols. Its committed clean-base Linux profile digest
    `sha256:f6962cd6c533197d19b3767e56595e21955d0514f89106bb2a37b0f118c77c33`
    passed locked install, validation, activation, the finite CLIP+LoRA child,
    rollback, and a second fresh clean-base process. The bounded 1,553-byte
    evidence has SHA-256
    `532322a065a3fce184fc8ba2bcabdde885226c4af91cc2be3764f76d9dd04c2f`
    and retained no managed state. The deterministic 117-workflow catalog,
    complete 1,543-test backend overlay with 3,355 subtests and three platform
    skips, Ruff E9/F, complete client gates plus the new exact recipe test, and
    unchanged 530,432-byte gzip ceiling pass. Remote real-weight
    memory/output safety/quality review and physical macOS execution remain
    pending independently; no weights or media were downloaded or retained.
  - [x] **Ideogram 4 source, terms, and admission-gate review:** backend
    `f09b06c` resolves the three official gated snapshots at immutable heads:
    `ideogram-ai/ideogram-4-nf4-diffusers@1874bc70267ba2c823a7239e1d70dd308c8d64dc`,
    `ideogram-ai/ideogram-4-nf4@f664347839e0a87bc495f5c9483cc0014b8e344e`,
    and `ideogram-ai/ideogram-4-fp8@ee79a7237b519f1402ceacf952f30c8a31ec5073`.
    Their anonymous metadata exposes four safetensors files and approximately
    16.10 GB, 16.10 GB, and 27.53 GB of visible weight bytes respectively,
    with no repository Python. These are deliberately recorded as visible
    sizes rather than exact artifact inventories: before gate acceptance the
    LFS identities are masked and the immutable model index and component
    configurations return HTTP 401. The package-owned pipeline, transformer,
    prompt-enhancer, scheduler, VAE, modular, and output source hashes, its
    2048-square / 48-step defaults, guidance schedule constraints,
    multiple-of-16 documented resolution range, callback/interrupt/offload
    surfaces, absent negative-prompt and safety-checker surfaces, and the
    optional prompt-enhancer boundary are sealed in
    `data/ideogram4-source-review.json`. No terms gate was accepted and no
    weight bytes or media were downloaded.

    Admission remains blocked. The exact June 3, 2026 Ideogram
    Non-Commercial Model Agreement was reviewed without acceptance: commercial
    use requires a separate agreement, hosted services and APIs count as
    distribution, downstream terms and notices attach, and the incorporated
    use policy makes suitable safety filters, human oversight, and disclosures
    product responsibilities while the package has no safety checker. MoDiff
    also lacks an authenticated exact-artifact receipt, a qualified
    NF4/BitsAndBytes CUDA loading recipe, and backend-owned guidance, resource,
    and safety bounds. Accordingly no runtime/download catalog, adapter,
    capability, graph, client, Auto, template, or Gallery surface is added.
    Task-scoped terms acceptance and legal/product approval, authenticated
    artifact review, remote heavy-hardware safety/quality qualification, and
    physical macOS execution remain independent gates.
  - [x] **JoyAI Image Edit and Edit Plus source admission:** backend
    `6d507eb` and client `b1cbde7` admit the exact public snapshots
    `jdopensource/JoyAI-Image-Edit-Diffusers@4b41fb25d961f37668750178ccbb380da326201c`
    and
    `jdopensource/JoyAI-Image-Edit-Plus-Diffusers@c2686460c7b64d8aa11bc4d0da423fb316b33f9e`
    through the generic Diffusers image facade. Both selected partitions are
    ungated, use twelve BF16 safetensors files, contain 16,263,675,968
    parameters, and require no repository Python or remote code. Their exact
    50,315,602,078-byte and 50,315,602,038-byte weight inventories, immutable
    repository metadata, component identities, and pinned package pipeline,
    image-processor, transformer, VAE, scheduler, and output source hashes are
    sealed in `data/joyimage-artifact-review.json` without downloading weight
    bytes. Both cards declare Apache-2.0 but link to an absent repository
    license file; the immutable upstream project's complete Apache-2.0 receipt
    is recorded while exact weight-snapshot license clarification remains
    explicit.

    The basic Expert-only route supports text-to-image and exactly one source
    image; Edit Plus supports one to five references and maps the generic input
    to the package's plural `images` argument. Both routes preserve the
    package's 1024-base aspect buckets, cap each output at 1,048,576 pixels and
    each side to 512-2048 in 32-pixel increments, cap text at the encoder's
    effective 2,048-token ceiling, and use guidance 4 with 40 or 30 steps.
    Generic generation now reports the actual decoded bucket dimensions rather
    than merely echoing the requested aspect hint. The package callbacks,
    interrupt surface, and sequential CPU offload are retained. The missing
    safety checker and unmeasured approximately 50.32 GB runtime keep Auto and
    Gallery disabled.

    The expanded Transformers/Diffusers symbol contract passed clean-base
    locked installation, validation, activation, a finite fresh-process
    CLIP+PEFT workload, rollback, and clean restoration on Linux x86-64 at
    profile digest
    `sha256:7fc2a03926b2a0d5fdee79c3178fe240707375d6b1ab2549bac4b2339220838c`.
    The 1,553-byte evidence has SHA-256
    `12af06fe33d3e36315783bb21f60fd5c1ea12c457622d743e94d05b935670604`
    and retained no managed state. Four canonical graphs bring the deterministic
    catalog to 121 supported workflows; runtime execution remains unqualified.
    The complete 1,555-test backend overlay with 3,398 subtests and three
    platform skips, Ruff E9/F, deterministic workflow verification, and the
    complete client check pass. The intentional profile surface measures
    530,495 compressed JavaScript bytes under a still-sub-KiB 531,456-byte
    ceiling. Remote real-weight memory/output safety/quality review, model-card
    license-file clarification, and physical macOS execution remain pending
    independently; no weights or media were downloaded or retained.
  - [x] Evaluate the remaining large image and cascaded families independently
    at the source/admission tier. Every standard image family in Appendix B now
    has an immutable source admission or an explicit fail-closed gate review;
    this closes inventory research only. The parent heavy-hardware item remains
    open for exact remote real-weight execution, output review, and physical
    macOS evidence, and no static receipt is treated as live proof.
- [x] Complete an explicit immutable-code security review and keep LLaDA2
  blocked. Backend `444152a` seals the public
  `inclusionAI/LLaDA2.1-mini` snapshot at
  `20e64e2ad21644d0e5248586ed9c942cdd45de0f`, its exact eight-file /
  32,513,130,952-byte safetensors inventory, configuration and index hashes,
  pinned Diffusers pipeline/scheduler source hashes, and both repository Python
  blobs. Full visual and static AST/string/import review found no evident file,
  network, process, unsafe-deserialization, dynamic-import, or dynamic-code
  primitive in those two immutable files. That is narrow evidence rather than
  a claim of runtime safety: importing the implementation mutates Transformers'
  process-global layer-normalization registry, and static review cannot bound
  dependencies or resource consumption.

  The official loading route still requires `trust_remote_code=True`, which
  would execute repository Python with all backend-process permissions. The
  pinned pipeline validates positive values but supplies no backend-owned upper
  bounds for prompt length, generation/block length, step counts, output size,
  or cooperative cancellation. This roadmap request is not fresh task-scoped
  operator authorization for that exact code. Accordingly no runtime/download
  catalog, capability, workflow, client, Auto, template, or Gallery surface was
  added. Explicit task-scoped authorization, bounded adapter controls, remote
  heavy-hardware execution, and physical macOS evidence remain independent
  gates.
- [x] Build the 30-minute video workflow only after chunk generation, checkpoint
  resume, deterministic stitching, audio mux, cancellation, and recovery pass
  independently.
  - [x] **Continuation boundary handoff:** backend `67010c7` makes the existing
    generic long-video planner require and bind an opening image for continuous
    LTX/Wan/FramePack plans, rejects unknown strategies, and gives the generic
    shot executor an optional previous-segment input. Only jobs carrying the
    explicit `uses_previous_last_frame` marker extract that segment's final
    frame as the next opening anchor; a missing prior segment fails before
    inference. A synthetic graph-level collection loop proves that iteration 2
    receives iteration 1's exact boundary through the normal carry contract.
    The focused video/loop matrix passes 129 tests and 216 subtests; the
    complete backend overlay passes 1,640 tests, 3,580 subtests, and three
    platform skips, with Ruff E9/F and package compatibility green. No model,
    media, or live inference was used. Durable process-restart checkpoints are
    closed by the later P6.57 slice; retained-asset stitching,
    audio mux/cancellation recovery, the actual 30-minute graph, remote
    execution, and physical macOS evidence remain independent gates.
  - [x] **In-process checkpoint, stitching, mux, and cancellation recovery:**
    backend `53e22f2` revalidates the generic implementation originally landed
    in `76bbafe` with stronger synthetic integration evidence. A collection
    loop is interrupted after committing segment 1, clears its node cache, and
    resumes at segment 2 without regenerating the completed result. Two real
    temporary eight-frame MP4 segments are retained, joined through the
    bounded-memory FFmpeg path with an exact two-frame transition into a
    14-frame/1.75-second result, then muxed with generated silent audio while
    preserving the video duration and frame count. The focused component gate
    passes 135 tests and 216 subtests; the complete backend overlay passes
    1,642 tests, 3,580 subtests, and three platform skips, with Ruff E9/F and
    package compatibility green. Test media existed only in the temporary test
    directory. Process-restart persistence is closed by the later P6.57 slice;
    the actual 30-minute graph, remote six-hour execution, asset publication,
    and physical macOS evidence remain pending.
  - [x] **Bounded 30-minute chunk planning:** backend `88b5d33` extends the
    generic planner's declared and enforced duration ceiling from 600 to 1,800
    seconds while retaining a separate 600-second ceiling for unqualified
    single-job FramePack output. A 16-FPS LTX continuation plan using
    five-second legal `8k+1` chunks and a 0.25-second overlap deterministically
    yields 374 jobs, 28,802 planned frames, and 1,800.125 seconds. Every job
    carries explicit FPS, width, height, steps, guidance, conditioning
    strength, negative prompt, seed, and continuation state. The planner caps
    jobs at 512 by default (10,000 hard maximum) and rejects duration, overlap,
    control, strategy, or job-count violations before inference. The focused
    component gate passes 137 tests and 216 subtests; the complete backend
    overlay passes 1,644 tests, 3,580 subtests, and three platform skips, with
    Ruff E9/F and package compatibility green. No model or media was used. The
    process-restart persistence is closed by the following slice; the
    executable 30-minute graph, remote six-hour run, asset publication, and
    physical macOS evidence remain pending.
  - [x] **Durable retained-segment restart recovery:** backend `d648417` and
    client `19620f9` add an explicit `durable` visual-loop contract for
    file-backed video segments. The Studio loop UI exposes the opt-in, the
    managed quality-video sequence seals it into its schema-v3 graph proof,
    and the API export carries it without making ordinary loops durable.
    Durable execution requires the exact `workflowTabId` plus `runInputHash`,
    accepts only bounded retained-video metadata whose file remains inside
    MoDiff's managed media directory, and atomically commits each successful
    iteration. A synthetic process-replacement test interrupts after segment
    1, constructs a new server with a different task ID, restores the exact
    input-scoped checkpoint, and executes only segment 2. Successful graph
    completion deletes checkpoint metadata; interrupted or failed exact-input
    runs retain it for recovery. The focused backend video/runtime matrix
    passes 205 tests and 246 subtests; the complete backend overlay passes
    1,646 tests, 3,580 subtests, and three platform skips, with Ruff E9/F and
    package compatibility green. The complete client gate, the focused
    controlled-workflow browser proof, and the 530,915-byte total gzip budget
    pass. Test files existed only under a temporary directory. The executable
    30-minute graph, remote six-hour run, asset publication, and physical macOS
    evidence remain pending.
  - [x] **Executable 30-minute qualification graph:** backend `479d495`, with
    schema-boundary fix `a9cf15c`, adds a qualification-only API graph
    template for the exact immutable
    `Lightricks/LTX-Video-0.9.8-13B-distilled` revision. It binds the BF16
    `LTXConditionPipeline`, eight-step/guidance-one recipe, opening image,
    1,800-second planner, 374-job durable continuation loop, retained pinned
    segment export, 0.25-second file-native join, and six-hour runtime ceiling
    in one app-submittable graph. Every connection resolves to a registered
    generic node and declared output.

    The companion materializer requires the local opening-image path and its
    portable app identifier to resolve to the same bytes, seals their digest
    into the input-scoped recovery identity, and accepts submission only with
    explicit long-run consent to a loopback app. Before `POST /graph` it also
    requires the app cache to report the exact immutable LTX revision complete
    and repair-free. Five direct graph/materializer tests plus the focused
    video/loop matrix passes 144 tests and 219 subtests; the complete backend
    overlay passes 1,651 tests, 3,583 subtests, and three platform skips, with
    Ruff E9/F and package compatibility green. A CLI smoke materialized the
    graph only to `/tmp`; it did not submit the graph, run inference, generate
    media, or download a model. The approximately six-hour remote run,
    output/safety/continuity review, asset publication, and physical macOS
    evidence remain pending.

### Phase 6 test and asset gate

- [x] No Phase 6 live run occurs on the current development machine.
- [x] Heavy integrations can merge contract-only while clearly Expert-only and
  `qualification_pending`.
- [x] Long-form component tests use synthetic/tiny segments.
- [ ] The approximately six-hour 30-minute-video qualification runs once as a
  scheduled release test after all component gates pass.
- [ ] Generated video and receipts are published through the remote asset
  workflow; no media is committed to Git.

## Per-integration admission checklist

Complete this research before implementing any pipeline or model entry:

- [ ] Confirm the class and workflow exist at the pinned Diffusers or approved
  Transformers revision.
- [ ] Record the exact call signature, required inputs, optional inputs, and
  return type.
- [ ] Identify the generic MoDiff task/media contract and necessary aliases.
- [ ] Inspect every model and auxiliary repository at an immutable revision.
- [ ] Record license, gating, remote-code, serialization, and redistribution
  constraints.
- [ ] Record stored artifact size and a conservative RAM/VRAM/disk envelope.
- [ ] Use only upstream-supported loaders, adapters, schedulers, and offload
  hooks.
- [ ] Define failure behavior and finite lower-resource retries.
- [ ] Define static, mocked/tiny, integrated, live, and asset evidence.
- [ ] Decide local-allowlisted or remote-only before downloading weights.

## Appendix A — Missing Modular classes

The current inventory contains 20 classes. All are present at the MoDiff pin and
all 20 are registered contract-only with reviewed no-weight workflow contracts.

Present in the current pin but not registered by MoDiff:

- [x] `AnimaModularPipeline`
- [x] `Cosmos3OmniModularPipeline`
- [x] `Cosmos3DistilledModularPipeline`
- [x] `ErnieImageModularPipeline`
- [x] `Flux2ModularPipeline`
- [x] `Flux2KleinBaseModularPipeline`
- [x] `HeliosModularPipeline`
- [x] `HeliosPyramidModularPipeline`
- [x] `HeliosPyramidDistilledModularPipeline`
- [x] `HunyuanVideo15ModularPipeline`
- [x] `Ideogram4ModularPipeline`
- [x] `LTXModularPipeline`
- [x] `StableDiffusion3ModularPipeline`
- [x] `Wan22ModularPipeline`
- [x] `Wan22Image2VideoModularPipeline`

Post-pin classes registered and reviewed by MoDiff:

- [x] `Krea2ModularPipeline`
- [x] `Krea2TurboModularPipeline`
- [x] `MiniMaxH3ModularPipeline`
- [x] `LTX2ModularPipeline`
- [x] `LTX25ModularPipeline`

## Appendix B — Missing standard pipeline families

This is a family inventory, not a requirement to create one node per family.

### Audio

- [x] `audioldm2`
- [x] `longcat_audio_dit`

### Text diffusion

- [x] `diffusion_gemma`
- [x] `llada2`

### 3D and perception

- [x] `shap_e`
- [x] `marigold`
- [x] `visualcloze`

### Video

- [x] `allegro`
- [x] `animatediff`
- [x] `anyflow`
- [x] `chronoedit`
- [x] `cogvideo`
- [x] `consisid`
- [x] `cosmos`
- [x] `easyanimate`
- [x] `helios`
- [x] `hunyuan_video1_5`
- [x] `kandinsky5`
- [x] `latte`
- [x] `lucy`
- [x] `mochi`
- [x] `motif_video`
- [x] `sana_video`
- [x] `skyreels_v2`
- [x] `stable_video_diffusion`

### Image, unconditional, and generic

- [x] `aura_flow`
- [x] `bria`
- [x] `bria_fibo`
- [x] `chroma`
- [x] `cogview3`
- [x] `cogview4`
- [x] `consistency_models`
- [x] `controlnet`
- [x] `controlnet_hunyuandit`
- [x] `controlnet_sd3`
- [x] `ddim`
- [x] `ddpm`
- [x] `deepfloyd_if`
- [x] `dit`
- [x] `dreamlite`
- [x] `ernie_image`
- [x] `glm_image`
- [x] `hidream_image`
- [x] `hunyuan_image`
- [x] `hunyuandit`
- [x] `ideogram4`
- [x] `joyimage`
- [x] `kandinsky`
- [x] `kandinsky2_2`
- [x] `kandinsky3`
- [x] `kolors`
- [x] `krea2`
- [x] `latent_consistency_models`
- [x] `latent_diffusion`
- [x] `ledits_pp`
- [x] `longcat_image`
- [x] `lumina`
- [x] `lumina2`
- [x] `nucleusmoe_image`
- [x] `omnigen`
- [x] `ovis_image`
- [x] `pag`
- [x] `pixart_alpha`
- [x] `prx`
- [x] `sana`
- [x] `stable_cascade`
- [x] `stable_diffusion`
- [x] `stable_diffusion_3`
- [x] `t2i_adapter`

## Completion ledger

Add references only after the corresponding evidence exists.

| Segment | Backend reference | Client reference | Live proof | Dataset revision | Status |
| --- | --- | --- | --- | --- | --- |
| P0.1 | `91c9a36` | `28b12b7` | Not required | Not required | Complete: exact-pair capability and stale-form execution checks are implemented and passed the recorded complete backend/client and browser gates. |
| P0.2 | `8fb2cb9`; corrected by `bf0af6b` | `c3e8a17`; corrected by `4cad1b2` | Not required | Not required | Complete: exact executable resource-plan targeting, bounded receipt binding, mixed/disconnected/zero-target rejection, and client fail-closed readiness/apply/run checks passed the complete backend/client and mocked-browser gates. The corrective pair makes the real schema-v2 backend publish the profile/schema fields already required by the client and binds them at admission; no model or asset execution was needed. |
| P0.3a.1 | `91c9a36` | `28b12b7` | Not required | Not required | Complete: registered Modular dynamic action safety and its backend/client gates are recorded in the paired implementation commits. |
| P0.3a.2 | `91c9a36` | `28b12b7` | Not required | Not required | Complete: safe declarative custom contract identity/preview and its backend/client/HTTP gates are recorded; executable custom admission remains deferred to P1.1. |
| P0.3b | `91c9a36` (revalidated at `8fb2cb9`) | `28b12b7`; Win32 checkpoint `d226c4b` (revalidated at `c3e8a17`) | Not required | Not required | Complete: P0.3b.1-.7 implementation, complete backend/client gates, reviewed Windows visual baselines, exact bundle mirror, and fresh HTTP smoke passed; live qualification is not part of this segment. |
| P0.3c | `e11263f`, `d81737e` | `aded6ca` (unchanged generic client contract revalidated) | Not required | Not required | Complete for the pinned upstream truth scope: all registered Modular action/component contracts and public modes are exact; SDXL base inpaint is contract-only; all 18 pinned SDXL workflows have exact generic state truth; Qwen/Layered and Wan split-state contracts close; standard image/video/audio adapters are registered only at their proved tier; and no client model-name branch was added. Multi-ControlNet/multiple-IP-Adapter expansion, templates/assets, and live qualification are distinct future gates, not evidence claimed by this phase. |
| P0.3d | `fd258d8` | `642ea9c` | Not required | Not required | Complete: backend-owned versioned Flux Schnell/Dev specifications, strict client parsing, generic graph materialization, exact proof/runtime receipt binding, complete backend/client/browser gates, byte-exact mirror verification, and local HTTP smoke passed. No model or media execution was required. |
| P0.3e | `96f70cb` (Flux Krea T2I), `a299d1d` (Flux Depth control-image), `14fef9f` (Flux Canny control-image), `6be23e7` (Flux Redux edit-image), `5f4d437` (Flux Kontext edit-image), `119c720` (Flux Kontext multi-reference edit), `544c54f` (Flux Fill inpaint), `634c485` (Flux Fill outpaint), `441cd00` (Flux2 Klein T2I), `ab3bd34` (Flux2 Klein edit-image), `4527764` (Flux2 Klein multi-reference edit), `276dd1f` (Wan TI2V text-to-video; corrected by `6983ce6`), `6983ce6` (Wan I2V image-to-video), `92cd1f5` (Wan 2.1 text-to-video), `93b1e17` (Wan 2.1 video-to-video), `09b1d4b` (Wan 2.1 color edit), `7736dd3` (LTX text-to-video), `7b8d5c1` (LTX image-to-video), `e83c760` (LTX video-to-video), `0bdc364` (LTX reference-to-video), `adcaf48` (ACE-Step text-to-audio), `5f6ffdc` (ACE-Step audio variation), `10b9b1c` (ACE-Step audio continuation), `60b87a1` (ACE-Step audio repaint), `de2160f` (Qwen Image Edit inpaint), `823357d` (Qwen Image Edit outpaint and bundle), `69a8561` (Wan VACE text-to-video; corrected by `2616014`, bundle `69247d0`), `0e9c3f2` (Wan VACE video inpaint and bundle), `b004af1` (Wan VACE video outpaint and bundle), `ed07f34` (Wan VACE control-to-video), `a4efd6c` (Z-Image Auto T2I), `6e40bab` (Qwen Image Auto T2I), `4596728` (Qwen Image Edit Modular), `0e7a8f1` (Qwen Image Edit Plus edit and multi-reference), `dd594ba` (Qwen Layered layer decomposition), `03c358b` (Qwen Image Control), `732e15c` (declarative loader-component outputs), `02afc25` (declarative layer-block allowlists), `7228c1f` (declarative Denoise image-latent dimensions), `b32241b` (generic video field overlay), `d3125dd` (Expert quantization resource policy), `f2ec7ac` (Expert MPS resource policy), `98f3841` (generic image and Modular field contracts), `fd514f7` (Expert quantization choices) | `80ac243` (Flux Krea T2I), `2de0c68` (Flux Depth control-image), `784e3c7` (Flux Canny control-image), `b709126` (Flux Redux edit-image), `8ae0dd9` (Flux Kontext edit-image), `d956a42` (Flux Kontext multi-reference edit), `38f8d81` (Flux Fill inpaint), `4c0bd05` (Flux Fill outpaint), `931621d` (Flux2 Klein T2I), `84e1d8f` (Flux2 Klein edit-image), `7180694` (Flux2 Klein multi-reference edit), `049addb` (Wan TI2V text-to-video; corrected by `60f4036`), `60f4036` (Wan I2V image-to-video), `0e359ce` (Wan 2.1 text-to-video), `651eeb3` (Wan 2.1 video-to-video), `2525937` (Wan 2.1 color edit), `9f2122f` (LTX text-to-video), `709ddd3` (LTX image-to-video), `8bd95e6` (LTX video-to-video), `cddd140` (LTX reference-to-video), `5f91ed9` (ACE-Step text-to-audio), `2a776c0` (ACE-Step audio variation), `7e69367` (ACE-Step audio continuation), `62dfe17` (ACE-Step audio repaint), `f28ff89` (Qwen Image Edit inpaint), `de2eba1` (Qwen Image Edit outpaint), `8f05541` (Wan VACE text-to-video; corrected by `4c9d40c`), `3f79ca2` (Wan VACE video inpaint), `fce224e` (Wan VACE video outpaint), `72ed446` (Wan VACE control-to-video), `77ceab9` (Z-Image Auto T2I), `531d4b9` (Qwen Image Auto T2I), `e8aab4e` (Qwen Image Edit Modular), `57a4072` (Qwen Image Edit Plus edit and multi-reference), `ff3f9c6` (Qwen Layered layer decomposition), `1102249` (Qwen Image Control), `947f7d9` (generic video field overlay), `5e8a1e7` (Expert quantization resource policy), `06ca70f` (Expert MPS resource policy), `c88e685` (generic image field switching browser proof), `a63d882` (image identity fallback removal), `5abfab9` (Expert quantization choices), `9cec2db` (exact installed-model loader identity), `d3ad700` (exact model-switch quantization retention), `229b5d1` (declarative low-memory presets), `a32b37a` (generic Modular readiness), `0c3a4c5` (exact restored quantization), `4afc515` (legacy graph fallback cleanup) | Not required | Not required | Complete: all 39 current execution-profile pairs and the shared loader, field, topology, readiness, and resource overlays are declarative and exact. The final residual audit removed active managed graph-construction model/pipeline switches while preserving explicit versioned template recipes, backend adapter normalization, and imported/manual Expert graph inference as declared boundaries. Complete client and 96/96 mocked-browser gates passed; no live model execution was required. |
| P0.3e Guider overlay | `51206e6` | `d1b2f88` | Not required | Not required | Complete: reviewed per-pipeline Guider choices, exact execution validation, scalar/multi-select dynamic option preservation, the complete backend/client gates, and the focused signal-relay mocked-browser contract passed. This closes the Guider portion of the parent P0.3e remaining-work summary. |
| P0.3e Scheduler overlay | `6779a19` | `140cab2` | Not required | Not required | Complete: pinned-upstream scheduler compatibility metadata, live-component and exact-constructor validation, complete backend/client gates, and the focused generic signal-relay mocked-browser contract passed. This closes the Scheduler portion of the parent P0.3e remaining-work summary. |
| P0.3e readiness overlay | `03c358b` (compatible exact-specification contract) | `7a02806` | Not required | Not required | Complete: readiness consumes the live managed loader identity or the unique authoritative execution specification, validates every exact role generically, and no longer routes the removed capability checks by model or pipeline name. Focused 67/67, complete client, exact browser, and final 88/88 mocked Studio gates passed; the bundle remained inside both limits. |
| P0.3e Modular readiness identity | Not required (uses the existing exact execution profile contract) | `a32b37a` | Not required | Not required | Complete: managed Run readiness identifies restored Modular graphs from generic managed roles and new graphs from the exact execution path, with no model-family branch. Focused 43/43, complete client, and final 96/96 mocked Studio gates passed; the bundle remained 429 bytes inside the stricter safety target. |
| P0.3e restored quantization admission | Not required (uses the existing exact execution profile contract) | `0c3a4c5` | Not required | Not required | Complete: bounded persisted quantization is admitted only by the exact selected execution profile, with no family filter. Focused 68/68, complete client, and final 96/96 mocked Studio gates passed; the bundle remained 395 bytes inside the stricter safety target. |
| P0.3e audio field overlay | `2a98856` | `fba496c` | Not required | Not required | Complete: reviewed audio pipeline/mode contracts publish the exact generic Generate field overlay; the backend rejects tampered overlays and the client no longer derives audio visibility from pipeline names. Complete backend/client and final 89/89 mocked Studio gates passed; no live audio execution was required. |
| P0.3e video field overlay | `b32241b` | `947f7d9` | Not required | Not required | Complete: every reviewed generic video adapter/mode owns its field visibility, required inputs, adapter controls, and strength binding; the exact backend action rejects stale contracts and the client no longer identifies LTX to choose the strength control. Complete backend/client and final 90/90 mocked Studio gates passed; no live video execution was required. |
| P0.3e Expert CUDA resource policy | `b1f514f` | `0259624` | Not required | Not required | Complete: exact Qwen execution profiles own the bounded dtype/offloaded/resident/quantized CUDA estimates, the client consumes only the policy attached to the selected exact specification, and no model-family fallback remains for these checks. Complete backend/client and final 91/91 mocked Studio gates passed; no live model execution was required. |
| P0.3e Expert quantization resource policy | `d3125dd` | `5e8a1e7` | Not required | Not required | Complete: exact Qwen execution profiles own the bounded Expert quantization/offload and generic-node component contract; the client strictly consumes it only through the selected exact specification, and direct/Modular readiness plus graph materialization no longer use a model-family quantization branch. Complete backend/client and final 91/91 mocked Studio gates passed; no live model execution was required. |
| P0.3e Expert MPS resource policy | `f2ec7ac` | `06ca70f` | Not required | Not required | Complete: exact reviewed execution profiles own the bounded Expert Apple MPS qualification and fallback advisory; the client strictly consumes it only through the selected exact specification, and no Qwen/Z/video family branch remains in MPS readiness. Complete backend/client and final 92/92 mocked Studio gates passed; no Apple Silicon or live model execution was required. |
| P0.3e image/Modular field overlay | `98f3841` | `c88e685` | Not required | Not required | Complete: exact generic image pipeline/mode contracts drive live field visibility, Modular generic nodes refresh from selected registry metadata, stale image overlays fail closed, and the complete backend/client/final 93/93 mocked Studio gates passed; no live model execution was required. |
| P0.3e image-path and Expert quantization-choice cleanup | `fd514f7` | `a63d882`, `5abfab9` | Not required | Not required | Complete: managed image topology and loader class now come only from the exact selected specification or existing managed binding; exact Qwen/Flux profiles own the bounded Expert quantization choices; controlled tab restore retains its execution-spec receipt. The complete backend/client gates and final 94/94 mocked Studio suite passed, with the bundle 259 bytes inside the stricter safety target. No live model execution was required. |
| P0.3e resource-path overlay | `8fb2cb9` (exact schema-v2 Auto target contract) | `16b7f12` | Not required | Not required | Complete: the client no longer guesses execution paths from Qwen or family identity before planning; exact selected backend candidates remain the only Auto path authority, and the complete 89/89 Studio gate passed. |
| P0.4 | `bf0af6b` (Auto schema/profile history binding), `f0ccd13` (optional-runtime receipt binding), `3a0b355` (specification-owned graph receipt binding), `e2a1bf2` (auxiliary-artifact receipt binding), `5cb785d` (executable controlled-LoRA history/cache receipt binding), `31cbc47` (current controlled-workflow artifact receipts), `a4efd6c` (Z-Image exact graph specification), `6e40bab` (Qwen Image exact graph specification), `4596728` (Qwen Image Edit Modular exact graph specification), `0e7a8f1` (Qwen Image Edit Plus exact graph specifications), `dd594ba` (Qwen Layered exact graph specification), `03c358b` (Qwen Image Control exact graph specification) | `12847d0`, `4cad1b2`, `0131ea7`, `453da03`, `54a610a` (exact controlled-artifact metadata and proof label), `77ceab9`, `531d4b9`, `e8aab4e`, `57a4072`, `ff3f9c6`, `1102249` | Not required | Not required | Complete for the current reviewed contract set: schema-v3 seals LoRA, sequence, upscaler, quality, soundtrack, and lyric/mux graph transformations; Auto candidates/history bind planner/profile/runtime/topology and all current executable auxiliary artifact receipts; every one of the 39 current execution-profile pairs has an exact backend-owned graph specification; stale, malformed, disconnected, or unreviewed receipt claims fail closed; and plan-time UI no longer presents base-only history as proof of controlled artifacts. Future controlled artifact kinds require a new reviewed receipt and qualification slice. |
| P0.5 | `4073711` (qualifier), `655baa6` (platform cutover), corrected by `1e95362`; clean-base topology revalidated by `e29fbf4` | `16046ab` (target-aware Setup status) | Windows x86-64 guarded live-model proof; Linux x86-64 clean-base/no-weight, supervised lifecycle, and production-cutover proof; macOS pending and base-delivered | Not required | Complete for qualified x86 targets: the exact six-target profile/delivery table enables explicit first-use install/activation only on Linux and Windows x86-64. Direct base dependencies remain only on macOS/ARM targets. A committed clean Linux CPU base contained 61 packages and none of the ten staged distributions; the exact overlay installed, validated, activated, passed the finite CLIP+LoRA child, rolled back, and restored a fresh clean base. The later clean-base full suite passed with 40 exact optional-upstream tests scoped to base-delivered or activated runtimes instead of reinstalling Transformers. A fresh worker exposed actionable status and rejected required execution with `optional_runtime_missing` before queueing. The 2026-08-14 focused source revalidation passes 94 tests and 929 subtests across target delivery, execution, qualification, guidance, and profile contracts. macOS and ARM rows remain explicitly non-actionable/base-delivered pending their own qualifier evidence. |
| P1.1 | `207d8f1`; actionable API message follow-up `5dc7313` | `c3e932c` | Not required | Not required | Complete: reviewed official component execution is bound to exact main/auxiliary Hub commits, an installed pinned pipeline/block pair, immediate identity revalidation, a private content-addressed metadata snapshot, and P0.5 runtime admission. Local sources remain preview-only and repository Python remains disabled without a future non-persistable task authorization. Complete backend/client and focused browser gates passed without model or asset execution. |
| P1.2 | `50dafa6` | `7c6bdbf` | Not required | Not required | Complete: the reproducible pinned snapshot normalizes Sequential, Auto, Loop, state, output, and component contracts; DynamicBlock exposes and executes only sidecar-carryable reviewed tasks; the client consumes the declarative task visibility contract generically. Complete backend/client and focused browser gates passed without model or asset execution. |
| P1.3 | `b48355b` | `f044594` | Not required | Not required | Complete: all three planned guiders use exact pinned official exports and constructor contracts; layer requirements, component compatibility, typed parameters, and backend-driven generic option signals passed complete backend/client and focused browser gates without weights. |
| P1.4 | `8e44eb5` | `5cb2998` | Not required | Not required | Complete: all 15 Modular classes present at the pin are split into image/video/multimodal contract-only batches with exact generated upstream workflow schemas and generic Expert visibility. They remain outside executable, Auto, template, Gallery, optional-runtime, and live-support registries; complete backend/client and focused browser gates passed without weights or assets. |
| P2.1 | `fa1884a` | `6f12230` | Not required | Not required | Complete: all 39 authoritative execution pairs publish a stable generic planning contract with exact execution-profile, loader, required-media, and terminal-output identity. Strict client parsing produces modality-generic skeletons and leaves every entry Gallery-hidden pending the separate qualification gate; complete backend/client gates passed without weights or assets. |
| P2.2a SDXL base text-to-image | `e5905f5` | `7581902` | Remote pending | Pending | Complete source slice: exact pinned planning graph and generic revision binding passed the complete backend/client gates; Auto and Gallery remain disabled pending live qualification and immutable assets. |
| P2.2b SDXL image-to-image | `5b03302` | `2bd15b2` | Remote pending | Pending | Complete source slice: one logical SDXL model selects an exact mode-specific img2img class and pinned planning graph; source-image, complete-suite, and bundle gates passed while Auto and Gallery remain disabled. |
| P2.2 task-contract planning foundation | `57316d9` | `2ec78a4` | Not required | Not required | Complete: the canonical generator falls back from curated recipes to exact backend task-template skeletons, and both pending SDXL base modes were regenerated without placeholder Gallery entries or model-specific client builders. |
| P2.2c SDXL inpaint | `a1faf3a` | `b9ea71e` | Remote pending | Pending | Complete source slice: exact mode-specific inpaint loader, immutable base revision, generic source/mask bindings, task-contract generation, complete suites, and bundle gate passed; Auto and Gallery remain disabled. |
| P2.2d FLUX.1-dev image-to-image | `b66439b` | `9a7280e` | Remote pending | Pending | Complete source slice: exact mode-specific img2img loader, immutable FLUX.1-dev revision, generic source-image binding, task-contract generation, complete suites, and bundle gate passed; Auto remains text-to-image-only and Gallery activation remains pending. |
| P2.2e FLUX.1-dev inpaint | `b679e4f` | `e8197c8` | Remote pending | Pending | Complete source slice: exact mode-specific inpaint loader, immutable FLUX.1-dev revision, generic source/mask bindings, task-contract generation, complete suites, and bundle gate passed; Auto remains text-to-image-only and Gallery activation remains pending. |
| P2.2f Z-Image Turbo image-to-image | `0d1d3cb` | `a586fce` | Remote pending | Pending | Complete source slice: exact standard img2img loader, immutable Z-Image Turbo revision, generic source-image binding, task-contract generation, complete suites, and bundle gate passed; Auto remains text-to-image-only and Gallery activation remains pending. |
| P2.2g Qwen-Image-2512 image-to-image | `8a4dbd8` | `538a79c` | Remote pending | Pending | Complete source slice: exact standard img2img loader, immutable Qwen-Image-2512 revision, generic source-image binding, reviewed Expert policy, task-contract generation, complete suites, and bundle gate passed; Auto remains unchanged and Gallery activation remains pending. |
| P2.2h Qwen-Image-2512 inpaint | `556be5f` | `bd8278f` | Remote pending | Pending | Complete source slice: exact standard inpaint loader, immutable Qwen-Image-2512 revision, generic source/mask bindings, reviewed Expert policy, task-contract generation, complete suites, and bundle gate passed; Auto remains unchanged, outpaint remains unadvertised, and Gallery activation remains pending. |
| P2.2 existing-image-path closure | `653168c` | `9862eea` | Remote pending | Pending | Complete: all 30 registered image pairs have deterministic canonical layouts; regeneration preserves catalog revisions and discovers base plus auxiliary Hub artifacts from each graph; focused backend integrity and complete client gates passed without weights or media. |
| P2.3 existing audio paths | `8e91284` | `c0170b2` | Remote pending for Stable Audio | Pending | Complete source slice: Stable Audio has an exact pinned generic task workflow; all five canonical audio pairs and both ACE LoRA variants verify deterministically; complete backend/client gates passed without weights or media. |
| P2.4 | `0ace2ae` | `8e25f5f` | Remote pending | Pending | Complete source slice: ten new exact short-video planning contracts and the existing Wan I2V/TI2V paths verify in the 70-workflow deterministic catalog; complete backend/client and focused browser gates passed without weights or media, while Auto and Gallery remain disabled pending P2.5. |
| P2.5 | Pending | Pending | Pending | Pending | Remote execution, review, publication, and activation have not started; clean-host campaign preflight is complete in P2.5a. |
| P2.5a Clean-host qualification campaign readiness | Not required | `8a93cf2` | Dry-run planning only; 76 live qualification receipts remain pending across six model-family batches | Pending | Missing local Auto history is correctly treated as no legacy evidence, ignored report directories initialize on clean hosts, and the complete client gate passes. The dry run submitted no graph and generated or published no media. |
| P2.5b Exact app-cache qualification readiness | Not required | `a76ee04` | Read-only live-app cache proof only; 76 live qualification receipts remain pending | Pending | All 76 selected jobs and 31 unique immutable model/LoRA receipts match complete, installed, repair-free app-cache entries. The loopback-only bounded preflight and complete client gate pass; no graph, inference, output, review, or publication occurred. |
| P2.5c Exact default-input qualification readiness | Not required | `d271a9f`, corrected by `7738537` | Read-only local-byte audit only; 76 live qualification receipts remain pending | Pending | The fail-closed campaign gate verifies selected Template Gallery defaults against their content-addressed bindings and asset-manifest size/hash receipts before browser or inference startup, checking both the authoring tree and the normal installer's durable backend `web/` payload. The runner uses the same installed-app fallback. The source checkout has none of the 50 required files (33,867,388 bytes), so 38 input-conditioned jobs are blocked and 38 input-free jobs are ready. No direct asset download, model deletion, graph, inference, output, review, or publication occurred. |
| P2.5d App-owned pinned Gallery materialization | `df71942` | `0fd0830` | Contract/unit/mocked-browser proof only; app activation, Gallery install, and 76 live qualification receipts remain pending | `b27198159c30d0c81aef397c188a7826866e5027` (`sha256:canonical-json:5ec869b755a6ce04a789d6835819da150493bfaef8a6bc1480f0274ba05bcab9` approved subset); payload not installed in this checkout | The app now exposes strict status, queue-aware plan, and explicit install/repair actions for the exact anonymous Dataset payload. It reserves download plus atomic staging bytes with active model reservations and a 64 GiB safety margin, hashes all 356 files / 480,430,370 bytes, and never deletes model caches. Complete backend/client gates and all 107 mocked Studio tests pass. The current old worker was intentionally not restarted while app-managed model downloads are active, so no Gallery POST/download occurred and the 38 conditioned jobs remain blocked until safe restart plus explicit in-app consent. |
| P2.5e Bounded parallel app downloads | `c313908` | Not required | Source/unit concurrency proof only; current old worker and live qualification remain pending | Not required | Two ordinary app snapshot transfers can now share the existing two-slot semaphore instead of serializing behind the process-global Xet lock. Repair is writer-exclusive and restores the prior Xet mode before normal transfers resume. Queue reservations, the 64 GiB reserve, immutable revisions, and no-deletion behavior are unchanged. Focused and complete backend gates pass. The active worker was not restarted, so its existing queue remains uninterrupted and this commit makes no current-live-transfer or output claim. |
| P2.5f Bounded app-owned Hub transport | `77ed298` | Not required | Source/unit transport, concurrency, and restoration proof only; current old worker and live qualification remain pending | Not required | App-owned model and Gallery snapshot payloads use standard Hub HTTP with bounded per-request timeouts/retries and share the existing two-transfer limit. Repair is writer-exclusive from cache preparation onward, completed blobs remain intact, exact global policy restoration is tested, and all admission/reserve/no-deletion behavior is unchanged. The focused 56-test matrix, five repeated race runs, complete 1,630-test backend gate, and static/package/preflight checks pass. The old active worker was not restarted, AuraFlow and the overnight queue were not interrupted, no active transport was switched, and no model/media/macOS qualification is claimed. |
| P3.4 Official Hugging Face library boundary | `91c9a36` | `28b12b7` | Not required | Not required | Complete: backend and client policy, contributor, security, dependency/runtime, and boundary-test contracts permit separately reviewed official Hugging Face libraries only behind MoDiff's local generic graph executor and immutable admission rules. The direct Transformers/PEFT base-dependency gap present at this checkpoint was later closed by P0.5's platform-scoped optional-runtime cutover. No model or media execution was required for this policy segment. |
| P3.1 | `80e4587` | `8f2a671` | Local cached CPU smoke passed; remote quality review pending | Pending | Complete source/live-smoke slice: the generic unconditional adapter, three immutable exact pairs, 73-workflow deterministic catalog, complete backend/client gates, and 102-case mocked Studio sweep passed. Auto and Gallery remain disabled pending remote output review and Dataset publication. |
| P3.2a Stable Diffusion 1.5 | `a0815b8` | `5a633a9` | Local cached CPU node smokes passed for text-to-image, img2img, and inpaint; remote quality review pending | Pending | Complete source/live-smoke slice: three exact generic pairs reuse one immutable safetensors base, the 76-workflow deterministic catalog and complete gates passed, and no generated media was retained. Auto and Gallery remain disabled pending remote output review and Dataset publication. |
| P3.2c Latent Consistency Model | `6a1b579` | `9f7b7da` | Local cached CPU node smoke passed at one step; remote quality review pending | Pending | Complete source/live-smoke slice: the exact immutable DreamShaper LCM pair uses the generic image nodes, the 77-workflow deterministic catalog and complete gates passed, and no generated media was retained. Auto and Gallery remain disabled pending remote output review and Dataset publication. |
| P3.2b Stable Diffusion 2.x | Pending: official repository access required | Pending | Pending | Pending | Deferred independently: 2026-08-14 app plans for the known immutable official base and inpaint revisions still returned repository-not-found/gated responses with unknown size. No download was submitted and no substitute or fabricated immutable revision was admitted. |
| P3.2d Perturbed-attention guidance | `662aa10` | `42c4dd6` | Local cached CPU node smoke passed at one step; remote quality review pending | Pending | Complete source/live-smoke slice: the exact PAG pair reuses the immutable SD1.5 safetensors base through generic image nodes, both PAG controls bind through the backend specification, the 78-workflow deterministic catalog and complete gates passed, and no generated media was retained. Auto and Gallery remain disabled pending remote output review and Dataset publication. |
| P3.3 Generic perception / Marigold depth | `957ab31` | `1802291` | Local cached CPU node smoke passed at one step; remote quality review pending | Pending | Complete source/live-smoke slice: the immutable Marigold Depth LCM pair uses a generic schema-versioned prediction-map boundary, the 79-workflow deterministic catalog and complete gates passed, and no generated media was retained. Normals, intrinsics, uncertainty, Auto, and Gallery remain disabled pending their separate qualification gates. |
| P3.5 Transformers speech-to-text | `82522ba` | `571facf` | Local cached CPU loader/action smoke passed with Transformers 5.14.1; remote spoken-fixture quality review pending | Pending | Complete source/live-smoke slice: two exact generic speech pairs use the immutable Whisper Tiny safetensors snapshot through the P0.5 optional runtime; the 81-workflow catalog and complete gates passed, and no fixture or output media was retained. Auto and Gallery remain disabled pending remote rights and quality review. |
| P4.1a SD1.5 ControlNet Canny | `539650a` | `785b43e` | Remote pending | Pending | Complete source slice: the immutable safetensors-only SD1.5/ControlNet assembly, exact generic Canny preprocessor, controlled artifact receipt, 82-workflow catalog, complete backend/client gates, and 106-case mocked Studio sweep passed. Auto and Gallery remain disabled pending remote output review. |
| P4.1b SD1.5 T2I Adapter | Deferred: reviewed official snapshot is legacy `.bin` only | Pending | Not attempted | Pending | Deferred independently under the safetensors-only auxiliary policy; no unsafe exception or community conversion was admitted. |
| P4.2a SDXL Turbo text-to-image | `fb49ed8` | `f893514` | Remote and physical macOS pending | Pending | Complete source slice: immutable fp16 safetensors loading, exact one-to-four-step guidance-zero contract, 83-workflow catalog, complete backend/client gates, and 106-case mocked Studio sweep passed. Auto and Gallery remain disabled pending license-surface and live output review. |
| P4.2b SDXL InstructPix2Pix image editing | `eb2a28e` | `b7ed626` | Remote and physical macOS pending | Pending | Complete source slice: immutable safetensors-only SDXL instruction editing, exact 768px/30-step/text-guidance-3/image-guidance-1.5 contract, 84-workflow catalog, complete backend/client gates, and 106-case mocked Studio sweep passed. Auto and Gallery remain disabled pending live output review. |
| P4.2c SDXL ControlNet Canny | `a4ae9ca` | `5440570` | Remote and physical macOS pending | Pending | Complete source slice: immutable fp16 safetensors base/component assembly, exact Canny preprocessor and 1024px/50-step/guidance-5/scale-0.5 contract, 85-workflow catalog, complete backend/client gates, shared-control coverage, and 106-case mocked Studio sweep passed. Auto and Gallery remain disabled pending live output review. |
| P4.2d SDXL T2I-Adapter Canny | `2d14051` | `b13d65d` | Remote and physical macOS pending | Pending | Complete source slice: immutable fp16 safetensors base/component assembly, exact Canny preprocessor and 1024px/30-step/guidance-7.5/scale-0.8 contract, 86-workflow catalog, complete backend/client gates, and shared-control regression coverage passed. Auto and Gallery remain disabled pending live output review. |
| P4.2e SDXL PAG text-to-image | `2a9c29b` | `379936e` | Remote and physical macOS pending | Pending | Complete source slice: immutable fp16 safetensors SDXL base, upstream PAG pipeline, exact 1024px/50-step/guidance-5/PAG-3/adaptive-0 contract, 87-workflow catalog, and complete backend/client gates passed. Auto and Gallery remain disabled pending live output review. |
| P4.2f SDXL PAG image-to-image and inpaint | `63f9075` | `c0f2e2b` | Remote and physical macOS pending | Pending | Complete source slice: immutable fp16 safetensors SDXL base, exact upstream PAG edit/inpaint classes, reviewed 1024px/50-step/guidance-5/strength-0.8/PAG-3/adaptive-0 contracts, 89-workflow catalog, and complete backend/client gates passed. Auto and Gallery remain disabled pending live output review. |
| P4.3 Sana/Sana Sprint and DreamLite admission | `7117c80`, `a56e9c0` | `c41d1c6`, `96f444b` (`2c99769` generator race fix) | Remote and physical macOS pending; DreamLite upstream per-step callback unavailable | Pending | Complete source slice: exact safe immutable artifacts, upstream classes, bounded distinct base/mobile recipes, seven canonical graphs, focused DreamLite gates, and complete client gate passed. DreamLite is Expert-only and CC-BY-NC-4.0; Auto and Gallery remain disabled. |
| P4.4 generic audio generation | `4d6a4d3` | `f0958e6` | Remote and physical macOS pending | Pending | Complete source slice: Stable Audio safe loading was revalidated and exact LongCat AudioDiT plus AudioLDM2 families now use immutable reviewed artifacts, bounded native-rate recipes, backend-owned declarative task contracts, two new canonical graphs, and complete gates. Auto and Gallery remain disabled. |
| P4.5 AudioLDM2 text-to-speech | Deferred: reviewed TTS snapshots are legacy `.bin` only | Pending | Not attempted | Pending | Deferred independently: the exact generic speech API is present at the pin, but both reviewed AudioLDM2 speech repositories require unsafe deserialization and no exception was approved. |
| P4.6 Shap-E rendered output | `21d819f` | `d684fc4` | Remote and physical macOS pending | Pending | Complete source slice: exact immutable official artifacts are assembled only from reviewed safe components, a bounded rendered-orbit boundary is sealed in the 95-workflow catalog, and complete backend/client gates passed. Unsafe renamed-renderer weights and mesh/export surfaces remain excluded. Auto and Gallery remain disabled. |
| P5.1 Stable Video Diffusion image-to-video | `260637d` | `d6e0eed` | Remote and physical macOS pending | Pending | Complete source slice: the exact gated official revision, safetensors-only artifact surface, license gate, documented offload/chunking recipe, bounded prompt-free image-conditioning contract, 96-workflow deterministic catalog, complete backend/client gates, and 106-case mocked Studio sweep passed. Auto and Gallery remain disabled; no weights or media were downloaded or retained. |
| P5.2 AnimateDiff and AnimateLCM | `75dde3c` | `220fb40` | Remote and physical macOS pending | Pending | Complete Expert-only source slice: immutable SD1.5 and motion revisions, exact safetensors-only adapters/LoRA, documented scheduler recipes, bounded 512px short-video execution, two sealed graphs in the 98-workflow catalog, complete backend/client gates, and the 106-case mocked Studio sweep passed. The motion repositories declare no weight license, so rights remain undetermined and require an explicit notice; Auto and Gallery remain disabled and no weights or media were downloaded. |
| P5.3a Motif Video evaluation | Deferred after immutable artifact/RAM review | Not required | Remote heavy-model review required | Pending | The official Apache-2.0 snapshot is safetensors-only, but its approximately 17.26 GB weight surface and native 121-frame 1280x736 recipe do not meet the smaller local-candidate premise. No executable or client surface was admitted and no weights or media were downloaded. |
| P5.3b CogVideoX-2B | `6501e22` | `00a3802` | Remote and physical macOS pending | Pending | Complete Expert-only source slice: exact Apache-2.0 safetensors artifact inventory, bounded native short-video contract, mandatory VAE tiling and model CPU offload, one sealed graph in the 99-workflow catalog, complete backend/client gates, and the 106-case mocked Studio sweep passed. Auto and Gallery remain disabled; no weights or media were downloaded. |
| P5 remaining short video | Pending | Pending | Remote pending | Pending | Existing Wan/LTX/LTX2/FramePack live qualification and any additional smaller candidates remain open. |
| P6.1 Diffusers pin update | `5ee9e1d` | Compatible client gate revalidated; no client change required | No live run required; remote model qualification remains pending | Not required | Complete isolated pin slice: the exact 73-commit delta was reviewed, existing no-weight Modular contracts remained structurally stable, required custom inputs were synchronized, the full backend suite passed at the proposed pin, and the repaired clean base is dependency-clean and preflight-ready. |
| P6.2 Krea2 Modular contracts | `1753384` | `ec2a349` | Contract-only; remote execution qualification pending | Not required | Both pinned classes are Expert-visible with exact, distinct base/Turbo contracts and fail closed before artifact resolution. The 28-contract snapshot, 1,313-test backend suite, complete client check, and 106-case mocked Studio sweep pass; no weights were downloaded. |
| P6.3 MiniMax H3 contracts and artifact review | `baf7271` | `947a2ef` | Contract-only; legal and remote heavy-hardware qualification pending | Not required | Three generic joint video/audio contracts, disjunctive FL2VA requirements, exact immutable partition/hash receipt, conditioner/scheduler/reference bounds, and an estimate-only resource envelope are sealed. The territory-restricted repository remains outside the runtime/download catalog with zero runnable modes. The 1,318-test backend suite, complete client check, and 106-case mocked Studio sweep pass; no weights or media were downloaded. |
| P6.4 LTX2/LTX2.5 contracts and source recipe review | `7e4f99b`, `b887aef` | `9dfcf62` | Contract-only; gated artifact indexes, license acceptance, remote heavy-hardware execution, and physical macOS qualification pending | Not required | All eight generic joint video/audio workflows and the distinct convolutional/diffusion decode contracts are sealed. The three official LTX-2.5 source recipes, exact sigma schedules, latent upsampler, duration head, explicit Gemma-4 enhancement, explicit NATTEN setup, and audio/video handoffs are recorded without exposing any runnable mode. Public immutable metadata covers 31 weight files / 163,896,920,128 bytes, but denied gated file access prevents exact partition selection; no weights or media were downloaded. |
| P6.5 HunyuanVideo 1.5 evaluation | `31cafc4` | Not required | Contract-only; territory/legal review and remote heavy-hardware execution pending | Not required | The existing two-workflow Modular contract, full official family, and immutable 480p T2V plus step-distilled I2V candidates are sealed with exact hashes, sizes, recipes, and estimate-only resource bounds. Conflicting territory language and additional commercial/distribution obligations keep all artifacts outside runtime and download catalogs. No weights or media were downloaded. |
| P6.6 Helios/Pyramid evaluation | `873f0ce` | Not required | Contract-only; immutable component-descriptor normalization and remote heavy-hardware execution pending | Not required | Base, Mid, and Distilled preserve their nine existing generic workflows. Exact full-repository and selected-partition receipts, distinct scheduler/guider recipes, chunk rounding, and estimate-only resource bounds are sealed. Upstream Modular indexes leave every component revision null, so no runtime or download entry was admitted. No weights or media were downloaded. |
| P6.7 Wan 2.2 A14B Modular evaluation | `011a70b` | Not required | Contract-only Modular path; remote fallback-assembly and heavy-hardware execution pending. Existing standard adapters remain graph-qualified/execution-pending. | Not required | Exact dual-expert T2V/I2V receipts, boundary-ratio fallback selection, workflow contracts, source recipes, and estimate-only resource bounds are sealed. No Modular index, new runtime/download catalog entry, weights, or media were added. |
| P6.8 classic LTX/LTX2 artifact evaluation | `0f96a92`, `474b83d` | Not required | Existing graph surfaces remain execution-pending; Modular paths, legal acceptance, and remote heavy-hardware execution remain pending | Not required | Exact full/selected inventories and source-contract receipts are sealed. The 2B family index can no longer silently replace the 13B Distilled profile. LTX-2's selected two-stage partition and license obligations are explicit. No weights or media were downloaded. |
| P6.9 Wan 2.1 14B Modular variants | `e4c2385` | Not required | Exact repository-scoped loader admission; remote heavy-hardware and physical macOS execution pending | Not required | T2V-14B and I2V-14B-720P join the already reviewed I2V-480P and FLF-720P variants under exact immutable catalog/index/component contracts. The focused clean-overlay matrix passes 100 tests plus 175 subtests. No new high-level mode, client branch, Auto/template/Gallery surface, weights, or media were added. |
| P6.10 LLaDA2 immutable-code security review | `444152a` | Not required | Static review only; explicit task-scoped authorization, bounded adapter controls, remote heavy-hardware execution, and physical macOS evidence pending | Not required | Exact remote-code blobs and eight-shard safetensors inventory are sealed. Static review found no prohibited primitive but did identify a process-global Transformers registry mutation and cannot prove runtime safety. `trust_remote_code` remains fail-closed; no runtime/download catalog or user-facing surface was admitted. |
| P6.11 DiffusionGemma artifact/source review | `42b609e` | Not required | No-weight API probe only; bounded generic diffusion-text contract, remote heavy-hardware execution, multimodal safety review, and physical macOS evidence pending | Not required | Exact official 11-shard / 51,647,701,024-byte safetensors inventory, Apache-2.0 rights, package-owned class/source hashes, 256-token/48-step entropy-bound recipe, callback support, and estimate-only resource envelope are sealed. The model remains remote-only and absent from runtime/download catalogs and user-facing capabilities. |
| P6.12 Stable Cascade artifact/source review | `b55983b` | Not required | Static artifact/source review only; license resolution, maintained package-owned pipeline support, remote heavy-hardware execution, and physical macOS evidence pending | Not required | Exact prior/decoder revisions, six-file / 13,728,020,596-byte selected bf16 partition, full inventories, source hashes, two-stage recipe, and estimate-only resource envelope are sealed. The noncommercial license, upstream deprecation, and unpinned connected-repository metadata keep the family outside runtime/download catalogs and user-facing capabilities. |
| P6.13 DeepFloyd IF artifact/source review | `8d45c9f` | Not required | Static artifact/source review and no-weight API probe only; authenticated gated-config review, backend-owned bounds, remote heavy-hardware execution, and physical macOS evidence pending | Not required | Three immutable stage revisions, 11-file / 27,326,661,461-byte repository-scoped selected surface, deduplicated weight size, source hashes, 64px-to-256px-to-1024px recipe, safety/watermark handoff, and estimate-only resource envelope are sealed. The gated noncommercial-research license keeps the family outside runtime/download catalogs and user-facing capabilities. |
| P6.14 PixArt Sigma 1024px source admission | `4fd1a66` (`8bca634` declarative-field fix) | `235c9d3` (`60b0269` exact-contract fix) | Remote real-weight, output safety/quality, and physical macOS execution pending | Not required | Exact public OpenRAIL++ revision, four-file / 21,827,405,446-byte safetensors inventory, package-owned pipeline/source hashes, bounded 1024px recipe, Expert-only remote workflow, and estimate-only resource envelope are sealed. The deterministic 104-workflow catalog is graph-qualified/runtime-unqualified; Auto and Gallery remain disabled and no weights or media were downloaded. |
| P6.15 AuraFlow v0.3 source admission | `8117b39` | `7d52469` (`c5e02ea` exact-contract fixture) | Remote real-weight, output safety/quality, and physical macOS execution pending | Not required | Exact public Apache-2.0 revision, four-file / 16,835,036,374-byte fp16 safetensors partition, package-owned pipeline/source hashes, bounded native 1536x768 recipe, Expert-only remote workflow, and estimate-only resource envelope are sealed. The deterministic 105-workflow catalog is graph-qualified/runtime-unqualified; Auto and Gallery remain disabled and no weights or media were downloaded. |
| P6.16 Bria 3.2 source and admission-gate review | `30a4667` | Not required | Static package/API and public gated-tree review only; authenticated exact artifact/license review, backend-owned precision and bounds, remote heavy-hardware execution, and physical macOS evidence pending | Not required | Package-owned classes, source hashes, call contract, public rounded safetensors observations, gated non-commercial terms, and HTTP-401 metadata limits are sealed without inventing an immutable artifact identity. The family remains outside runtime/download catalogs and all user-facing surfaces; no weights or media were downloaded. |
| P6.17 Bria FIBO generation/edit source and admission-gate review | `a826a5e` | Not required | Static package/API/custom-code review only; license acceptance, authenticated config/license review, explicit remote-code authorization, backend-owned structured-JSON/device/model pinning and bounds, remote heavy-hardware execution, and physical macOS evidence pending | Not required | Exact generation/edit heads, current and archived safetensors partitions, package source hashes, structured generation/edit/inpaint contracts, promptifier code revisions, and nested VLM inventories are sealed. Gated non-commercial weights plus revision-unbound CUDA-only custom promptifiers keep the family outside runtime/download catalogs and all user-facing surfaces; no weights or media were downloaded. |
| P6.18 Chroma1-HD text-to-image source admission | `5d3bc8f` | `86cbd92` | Remote real-weight, output safety/quality, and physical macOS execution pending | Not required | Exact public Apache-2.0 revision, five-file / 27,492,403,238-byte selected bfloat16 Diffusers partition, excluded duplicate single-file artifact, immutable metadata and package source hashes, bounded 1024px recipe, Expert-only remote workflow, and estimate-only resource envelope are sealed. The deterministic 106-workflow catalog is graph-qualified/runtime-unqualified; image-to-image, Auto, and Gallery remain disabled, upstream declares no safety alignment, and no weights or media were downloaded. |
| P6.19 CogView3 Plus 3B text-to-image source admission | `f74c806` | `a4d0b99` | Remote real-weight, output safety/quality, license-file clarification, and physical macOS execution pending | Not required | Exact public revision, seven-file / 25,559,227,422-byte bfloat16 Diffusers partition, immutable metadata and package source hashes, bounded 512-2048px recipe, Expert-only remote workflow, and estimate-only A100 resource envelope are sealed. Model-card metadata declares Apache-2.0, but its linked `LICENSE.md` is absent from the immutable tree. The deterministic 107-workflow catalog is graph-qualified/runtime-unqualified; Auto and Gallery remain disabled and no weights or media were downloaded. |
| P6.20 CogView4 6B text-to-image source admission | `495d07d` | `9822baa` | Remote real-weight, output safety/quality, and physical macOS execution pending | Not required | Exact public Apache-2.0 revision, eight-file / 31,108,954,670-byte bfloat16 Diffusers partition, immutable metadata and package source hashes, bounded 512-2048px/2^21-pixel recipe, Expert-only remote workflow, and estimate-only A100 batch-four resource envelope are sealed. The signature/docstring token discrepancy and contradictory 1920x1280 memory row are explicit. The deterministic 108-workflow catalog is graph-qualified/runtime-unqualified; Auto and Gallery remain disabled and no weights or media were downloaded. |
| P6.21 VisualCloze source and admission-gate review | `0f4d3c4` | Not required | Static artifact/source review only; generic visual-context-matrix contract, backend-owned bounds, license-file clarification, remote heavy-hardware execution, live output review, and physical macOS evidence pending | Not required | Exact public 384px and 512px revisions, two seven-file / 33,743,379,958-byte bfloat16 safetensors inventories, immutable metadata and package source hashes, nested matrix/generation/SDEdit contracts, and estimate-only resource envelopes are sealed. The legacy `.pth` LoRA artifacts are explicitly excluded. Contract mismatch and package validation gaps keep the family outside runtime/download catalogs and all user-facing surfaces; no weights or media were downloaded. |
| P6.22 Allegro text-to-video source admission | `6488462` | `05a2e15` | Remote real-weight, output safety/quality, license-file clarification, and physical macOS execution pending | Not required | Exact public revision, six-file / 25,293,069,108-byte bfloat16 Diffusers partition, excluded duplicate unsafe `.bin` partition, immutable metadata and package source hashes, bounded native 1280x720/88-frame recipe, float32 tiled VAE, Expert-only remote workflow, and conservative resource envelope are sealed. The deterministic 109-workflow catalog is graph-qualified/runtime-unqualified; Auto and Gallery remain disabled and no weights or media were downloaded. |
| P6.23 AnyFlow source and admission-gate review | `09d3c98` | Not required | Static artifact/source review only; noncommercial-license legal approval, backend-owned bounds, remote heavy-hardware execution, live output review, and physical macOS evidence pending | Not required | All four exact public bidirectional/FAR 1.3B/14B revisions, their seven- or nine-file / 26.07-51.87 GB bfloat16 safetensors inventories, complete identical license files, immutable metadata and package source hashes, native T2V/I2V/V2V contracts, FAR chunking, and estimate-only resource envelopes are sealed. The restrictive NVIDIA license and stale custom-code model-card API examples keep the family outside runtime/download catalogs and all user-facing surfaces; no weights or media were downloaded. |
| P6.24 ChronoEdit source and admission-gate review | `70640ae` | Not required | Static artifact/source review only; governing-license approval, complete safe guardrail integration, generic edit/reasoning contract, backend-owned bounds, remote heavy-hardware execution, live output review, and physical macOS evidence pending | Not required | The exact public 21-file / 90,075,130,404-byte safetensors core, three optional LoRA artifacts, immutable metadata and package source hashes, native image-edit/temporal-reasoning recipes, stale model-index identities, and measured upstream offload figures are sealed. The external governing terms' guardrail condition, package pipeline's missing safety checker, and bundled unsafe `.pth`/`.pt` guardrail artifacts keep the family outside runtime/download catalogs and all user-facing surfaces; no weights or media were downloaded. |
| P6.25 ConsisID source and admission-gate review | `a52c7ec` | Not required | Static artifact/source review only; safe cross-platform face stack, biometric privacy/consent controls, generic identity-video contract, backend-owned bounds, license-file clarification, remote heavy-hardware execution, live identity/safety review, and physical macOS evidence pending | Not required | The exact public five-file / 22,821,396,692-byte safetensors generator and eight-artifact / 1,446,798,634-byte required identity stack, immutable metadata and package source hashes, native 720x480/49-frame recipe, and measured upstream memory figures are sealed. Required unsafe face weights, CUDA-only ONNX providers, weak identity-input validation, and the unavailable second documented checkpoint keep the family outside runtime/download catalogs and all user-facing surfaces; no weights or media were downloaded. |
| P6.26 Latte text-to-video source admission | `180917e` | `847ed6c` | Remote real-weight, output safety/quality, license-file clarification, and physical macOS execution pending | Not required | Exact public revision, six-file / 23,614,979,636-byte safetensors partition, excluded unsafe legacy `.pt` checkpoint and unreferenced optional temporal VAE, immutable metadata and package source hashes, bounded native 512x512/16-frame recipe, Expert-only remote workflow, and estimate-only resource envelope are sealed. The deterministic 110-workflow catalog is graph-qualified/runtime-unqualified; Auto and Gallery remain disabled and no weights or media were downloaded. |
| P6.27 Lucy Edit source and admission-gate review | `6ab1033` | Not required | Static artifact/source review only; commercial-license and legal product approval, immutable governing terms, backend-owned bounds, remote heavy-hardware execution, live output review, and physical macOS evidence pending | Not required | The exact public five-file / 34,182,223,896-byte float32 safetensors inventory, immutable metadata, external license-document receipt, package source hashes, and native 832x480/81-frame edit recipe are sealed. The non-commercial/non-production license defines hosted remote access as distribution, and the package does not bind `num_frames` to input-video length, so the family remains outside runtime/download catalogs and all user-facing surfaces; no weights or media were downloaded. |
| P6.28 Mochi 1 Preview text-to-video source admission | `fb3e39f` | `c0afea5` | Remote real-weight, output safety/quality, license-file clarification, and physical macOS execution pending | Not required | Exact public revision, eight-file / 40,024,303,350-byte selected safetensors partition, excluded duplicate original-format, unindexed T5, and float32 partitions, immutable metadata and package source hashes, bounded native 848x480/31-frame recipe, explicit indexed T5 preload, mandatory VAE tiling, Expert-only remote workflow, and estimate-only resource envelope are sealed. The deterministic 111-workflow catalog is graph-qualified/runtime-unqualified; Auto and Gallery remain disabled and no weights or media were downloaded. |
| P6.29 SANA-Video 2B 480p text/image-to-video source admission | `081a083` | `6ed67bf` | Remote real-weight, output safety/quality, and physical macOS execution pending | Not required | Exact public Apache-2.0 revision, five-file / 13,963,813,420-byte mixed-precision safetensors partition, excluded unsafe original-format, distinct 720p, and duplicate-heavy LongLive repositories, immutable metadata and package source hashes, bounded native 832x480/81-frame T2V and I2V recipes, FP32 tiled Wan VAE, Expert-only remote workflows, and estimate-only resource envelope are sealed. The deterministic 113-workflow catalog is graph-qualified/runtime-unqualified; Auto and Gallery remain disabled and no weights or media were downloaded. |
| P6.30 Hunyuan-DiT v1.2 ControlNet Canny source admission | `72185d0` | `f134f98` | Remote real-weight memory/output safety/quality and physical macOS execution pending | Not required | Exact public distilled-base and Canny revisions, six-file / 17,399,623,404-byte float32 safetensors inventory, optional exact Depth/Pose substitutions, immutable Tencent license receipt and acknowledgement, metadata and package source hashes, bounded native 1024px/50-step/guidance-6/scale-1 Canny recipe, Expert-only remote workflow, and estimate-only resource envelope are sealed. The deterministic 114-workflow catalog is graph-qualified/runtime-unqualified; the missing safety checker keeps Auto and Gallery disabled, and no weights or media were downloaded. |
| P6.31 Stable Diffusion 3 ControlNet source and admission-gate review | `0f924dc` | Not required | Static artifact/source review only; authenticated base-config review, auxiliary-weight rights resolution, legal product approval, backend-owned optional-runtime/bounds, remote heavy-hardware output review, and physical macOS evidence pending | Not required | Exact gated base plus public Canny, Tile, and inpainting revisions; three safetensors-only assembly receipts; immutable license/metadata and package source hashes; native 1024px recipes; and estimate-only resource envelopes are sealed. Base access/license restrictions, undeclared InstantX weight rights, ambiguous inpainting derivative terms, and absent safety guardrails keep the family outside all runtime/download and user-facing surfaces; no weights or media were downloaded. |
| P6.32 DiT source and admission-gate review | `17ccba9` | Not required | Static artifact/source review only; safe official artifacts, commercial product rights, backend-owned loader/bounds/cancellation, remote heavy-hardware output review, and physical macOS evidence pending | Not required | The only two exact Facebook 256px/512px revisions, immutable metadata and package source hashes, four legacy weight identities, fixed ImageNet class-label contracts, and estimate-only resource envelopes are sealed. Legacy pickle-only serialization, CC BY-NC licensing without a bundled license file, absent safety guardrails, and absent cooperative cancellation keep the family outside all runtime/download and user-facing surfaces; no weights or media were downloaded. |
| P6.33 ERNIE Image Turbo text-to-image source admission | `a0b07c8` | `0527a66` | Remote real-weight memory/output safety/quality and physical macOS execution pending | Not required | Exact public Apache-2.0 revision, five-file / 31,596,733,630-byte predominantly BF16 safetensors inventory, immutable metadata and package/Transformers source hashes, fixed 1024px/8-step/guidance-1/prompt-enhanced recipe, Expert-only remote workflow, and estimate-only resource envelope are sealed. The revised optional-runtime symbol contract passed a clean-base locked install/activation/workload/rollback qualification. The deterministic 115-workflow catalog is graph-qualified/runtime-unqualified; the missing safety checker keeps Auto and Gallery disabled, and no weights or media were downloaded. |
| P6.34 GLM-Image text-to-image source admission | `c338824` | `d42a15c` | Remote real-weight memory/output safety/quality, bundled-license clarification, and physical macOS execution pending | Not required | Exact public MIT-declared revision, incorporated Apache-2.0 tokenizer terms, nine-file / 35,765,307,854-byte mixed BF16/FP32 safetensors inventory, immutable metadata and package/Transformers source hashes, fixed 1024px/50-step/guidance-1.5 recipe, Expert-only remote workflow, and estimate-only resource envelope are sealed. The revised optional-runtime symbol contract passed a clean-base locked install/activation/workload/rollback qualification. The deterministic 116-workflow catalog is graph-qualified/runtime-unqualified; the missing negative-prompt API and safety checker keep Auto and Gallery disabled, and no weights or media were downloaded. |
| P6.35 HiDream-I1 source and admission-gate review | `0685ee5` | Not required | Authenticated Llama 3.1 terms/artifact review, composite license receipt, backend-owned external encoder assembly/bounds, remote heavy-hardware output review, and physical macOS execution pending | Not required | Exact public Full/Dev/Fast revisions, three 12-file / approximately 47.18 GB safetensors partitions, shared and variant-specific immutable weight identities, package source hashes, official 50/28/16-step recipes, callbacks, and estimate-only 63.24 GB composite runtime surface are sealed. Every public snapshot omits the required Llama tokenizer/encoder; its manual gate masks artifact identities before acceptance. No runtime/download or user-facing surface is added, and no weights or media were downloaded. |
| P6.36 HunyuanImage 2.1 source and territory-gate review | `4987495` | Not required | Legal territory/distribution approval, product territory enforcement, immutable composite terms receipt, remote heavy-hardware output review, and physical macOS execution pending | Not required | Exact public package-owned conversion and governing upstream revisions, ten-file / 53,124,614,990-byte BF16 safetensors inventory, immutable license/notice/config and package source hashes, 2K/50-step/APG-3.5 first-stage recipe, callbacks, and estimate-only resource envelope are sealed. Express EU/UK/South-Korea exclusions keep the family contract-only and outside every runtime/download and user-facing surface; no weights or media were downloaded. |
| P6.37 Hunyuan-DiT v1.2 Distilled standalone source admission | `1e97362` | `e6e306f` | Remote real-weight memory/output safety/quality and physical macOS execution pending | Not required | The existing exact public distilled base is now a standalone five-file / 14,422,655,700-byte float32 safetensors source with an independently bounded 1024px/25-step/guidance-5 Expert workflow and immutable Tencent terms acknowledgement. The expanded HunyuanDiT optional-runtime symbol surface passed clean-base locked installation, activation, finite workload, rollback, and clean restoration. The deterministic 117-workflow catalog is graph-qualified/runtime-unqualified; the missing safety checker keeps Auto and Gallery disabled, and no weights or media were downloaded. |
| P6.38 Ideogram 4 source, terms, and admission-gate review | `f09b06c` | Not required | Static gated source/terms review only; task-scoped terms acceptance, authenticated artifact review, commercial agreement/legal approval, remote heavy-hardware output review, and physical macOS evidence pending | Not required | Three exact gated official heads, anonymous visible weight sizes, immutable June 3, 2026 noncommercial terms, and pinned package pipeline/transformer/prompt-enhancer/scheduler/VAE/modular source hashes are sealed without accepting the gate or downloading weights. Masked LFS identities, HTTP-401 configs, commercial/hosted-distribution restrictions, and absent package safety guardrails keep the family outside every runtime/download and user-facing surface. |
| P6.39 JoyAI Image Edit and Edit Plus source admission | `6d507eb` | `b1cbde7` | Remote real-weight memory/output safety/quality, model-card license-file clarification, and physical macOS execution pending | Not required | Exact public Apache-2.0-declared basic and Plus revisions, two twelve-file / approximately 50.32 GB BF16 safetensors partitions, immutable upstream Apache receipt, metadata and package/Transformers source hashes, bounded 1024-base bucket recipes, one- and five-reference generic edit contracts, and four Expert-only remote workflows are sealed. The revised optional-runtime symbol contract passed clean-base locked install/activation/workload/rollback qualification. The deterministic 121-workflow catalog is graph-qualified/runtime-unqualified; the missing safety checker keeps Auto and Gallery disabled, and no weights or media were downloaded. |
| P6.40 Kandinsky 2.1 source and admission-gate review | `c04f151` | Not required | Static artifact/source review only; exact connected-prior binding, full-job cancellation, backend-owned composite bounds, output guardrails, model-snapshot license clarification, remote heavy-hardware output review, and physical macOS evidence pending | Not required | Three exact public decoder/prior/inpaint revisions, their 13.23 GB composite safetensors-only surfaces, immutable metadata and package source hashes, upstream Apache receipt, three combined-mode contracts, and bounded header evidence are sealed. The package reuses the decoder revision on the distinct prior repository or downloads its moving branch, while the prior stage has no callback; the family remains outside every runtime/download and user-facing surface, and no full weights or media were downloaded. |
| P6.41 Kandinsky 2.2 source and admission-gate review | `75bb0ac` | Not required | Static artifact/source review only; exact connected-prior binding, backend-owned two-stage assembly/bounds, safe ControlNet/refiner artifacts, output guardrails, snapshot license clarification, remote heavy-hardware output review, and physical macOS evidence pending | Not required | Five exact official repositories, safe 15.86 GB decoder/prior composite surfaces, immutable metadata and package source hashes, two-stage callback contracts, and upstream Apache receipt are sealed. The connected loader remains revision-inexact; official depth-ControlNet and refiner snapshots are legacy `.bin`-only, and the refiner names the 2.1 pipeline without card/license metadata. No runtime/download or user-facing surface was added, and no full weights or media were downloaded. |
| P6.42 Kandinsky 3 text-to-image and image-edit source admission | `b85b073` | `3007bf3` | Remote real-weight memory/output safety/quality, model-snapshot license-file clarification, and physical macOS execution pending | Not required | Exact public Apache-2.0-declared revision, seven-file / 28,390,829,958-byte fp16 safetensors partition, immutable upstream Apache receipt, metadata and package/Transformers source hashes, bounded single-stage 1024px/25-step/guidance-3 routes, and two Expert-only remote workflows are sealed. The revised optional-runtime symbol contract passed clean-base locked install/activation/workload/rollback qualification. The deterministic 123-workflow catalog is graph-qualified/runtime-unqualified; the missing safety checker keeps Auto and Gallery disabled. Its exact snapshot is queued through the app after aggregate free-space reservation preflight, without deleting older models; no media has been generated. |
| P6.43 Kolors source and custom-license gate review | `3788fe8` | Not required | Static artifact/source/license review only; task-scoped license acceptance, commercial registration/legal approval, downstream restriction implementation, backend-owned bounds, remote heavy-hardware output review, and physical macOS execution pending | Not required | The exact public five-file / 17,813,668,046-byte fp16 safetensors partition, immutable metadata/license and package source hashes, and two package-owned 1024px routes are sealed. The custom model agreement conflicts with the Apache-2.0 presentation, purports to trigger on use/access, propagates restrictions, and requires separate authorization for cloud vendors or licensees over 100M monthly users. No runtime/download or user-facing surface was added, and no weights or media were downloaded. |
| P6.44 Latent Diffusion source and admission-gate review | `79db3b8` | Not required | Static artifact/source review only; safe official artifacts, cooperative full-job cancellation, backend-owned resource/output bounds, immutable model-license receipt, remote output review, and physical macOS execution pending | Not required | The exact public three-file / 6,152,286,891-byte legacy weight partition, immutable metadata and package source hashes, and the package's 256px text-to-image contract are sealed without fetching weight bytes. The snapshot has only executable pickle `.bin` model components, while the package exposes no callback, interrupt flag, safety checker, or upper resource bounds. No runtime/download or user-facing surface was added. |
| P6.45 LEDITS++ source and admission-gate review | `7bddd15` | Not required | Static package/source review over existing exact SD 1.5 and SDXL bases; full-job cancellation, request-state isolation, backend-owned resource/input bounds, generic multi-prompt editing, SDXL guardrails, remote output review, and physical macOS execution pending | Not required | The two package-owned source identities and stateful invert-then-edit contracts are sealed against MoDiff's existing exact base snapshots. The mandatory inversion phase has no callback or interrupt check and stores request state on the pipeline instance. LEDITS++ needs no distinct model snapshot, so no new weight bytes or family-specific app download were required, and no runtime or user-facing surface was added. |
| P6.46 LongCat Image generation and edit source admission | `212997c` | `d1e5d7e` | Remote real-weight memory/output safety/quality, model-snapshot license-file clarification, and physical macOS execution pending | Not required | Two exact public Apache-2.0-declared revisions, two seven-file / approximately 29.29 GB safetensors partitions, immutable upstream Apache receipt, metadata and package/Transformers source hashes, bounded 1024-base generation and single-image edit recipes, and two Expert-only remote workflows are sealed. The expanded optional-runtime symbol contract passed clean-base locked install/activation/workload/rollback qualification. The deterministic 125-workflow catalog is graph-qualified/runtime-unqualified; the missing safety checker keeps Auto and Gallery disabled. Both exact snapshots are queued through the app after aggregate free-space reservation preflight, without deleting older models; no media has been generated. |
| P6.47 Lumina Next and Lumina Image 2.0 source admission | `faad48b` | `5b2f3db` | Remote real-weight memory/output safety/quality, model-snapshot license-file clarification, and physical macOS execution pending | Not required | Two exact public Apache-2.0-declared revisions, four-file / 8.86 GB and six-file / 21.23 GB safetensors partitions, immutable upstream MIT/Apache receipts, metadata and package/Transformers source hashes, and two bounded Expert-only text-to-image workflows are sealed. Lumina 2.0's exact app allowlist excludes both legacy pickle artifacts. The expanded optional-runtime symbol contract passed clean-base locked install/activation/workload/rollback qualification. The deterministic 127-workflow catalog is graph-qualified/runtime-unqualified; missing safety checkers keep Auto and Gallery disabled. Fresh exact app plans now report Lumina Next's 16-file / 8,878,690,722-byte and Lumina Image 2.0's 17-file / 21,253,711,141-byte bounded selections complete at their immutable revisions. Both app cache entries are installed, complete, and repair-free. No older model was deleted and no media has been generated. |
| P6.48 OmniGen v1 generation and reference-edit source admission | `11d5c16` | `86500ec` | Remote real-weight memory/output safety/quality, model-snapshot license-file clarification, and physical macOS execution pending | Not required | Exact public MIT-declared revision, two-file / 8.09 GB safetensors partition, immutable upstream MIT receipt, metadata and package/Transformers source hashes, backend-generated ordered reference placeholders, bounded one- and three-reference recipes, and three Expert-only remote workflows are sealed. The expanded optional-runtime symbol contract passed clean-base locked install/activation/workload/rollback qualification. The deterministic 130-workflow catalog is graph-qualified/runtime-unqualified; the missing safety checker keeps Auto and Gallery disabled. A fresh exact app plan reports all 11 files / 8,088,956,424 selected bytes complete at the immutable revision; the cache entry is installed, complete, and repair-free. No older model was deleted and no media has been generated. |
| P6.49 Ovis Image 7B text-to-image source admission | `df2fa98` | `ab47b67` | Remote real-weight memory/output safety/quality and physical macOS execution pending | Not required | Exact public Apache-2.0 revision, a 21-file / 21.81 GB Diffusers-only selection, five immutable safetensors weights, bundled LICENSE/NOTICE receipts, and explicit exclusion of duplicate native checkpoints plus the Python-bearing Ovis2.5 subtree are sealed. The bounded 1024px/50-step/guidance-5 Expert workflow retains package cancellation and offload hooks. The expanded optional-runtime symbol contract passed clean-base locked install/activation/workload/rollback qualification. The deterministic 131-workflow catalog is graph-qualified/runtime-unqualified; the missing safety checker keeps Auto and Gallery disabled. A fresh exact app plan reports all 21 files / 21,806,937,901 selected bytes complete at the immutable revision; the cache entry is installed, complete, and repair-free. No older model was deleted and no media has been generated. |
| P6.50 PRX 512 SFT text-to-image source admission | `012b101` | `9c320a4` | Remote real-weight memory/output safety/quality and physical macOS execution pending | Not required | Exact public Apache-2.0 revision plus incorporated T5-Gemma terms, a complete 19-file / 15.51 GB Python-free snapshot, five immutable safetensors weights, and pinned package/runtime receipts are sealed. The bounded native 512px/28-step/guidance-5 Expert workflow maps the generic token limit to PRX's exact argument and retains step-boundary cancellation. The expanded optional-runtime symbol contract passed clean-base locked install/activation/workload/rollback qualification. The deterministic 132-workflow catalog is graph-qualified/runtime-unqualified; the missing safety checker keeps Auto and Gallery disabled. A fresh exact app plan reports all 19 files / 15,514,188,109 selected bytes complete at the immutable revision; the cache entry is installed, complete, and repair-free. No older model was deleted and no media has been generated. |
| P6.51 Nucleus Image 17B MoE text-to-image source admission | `86cc756` | `8b4f002` | Model-snapshot license-file clarification, app storage capacity, remote real-weight memory/output safety/quality, and physical macOS execution pending | Not required | Exact public Apache-2.0-declared revision, complete 38-file / 51.66 GB Python-free snapshot, 12 immutable safetensors weights, and pinned package/runtime receipts are sealed. The bounded seven-bucket 1024px/50-step/guidance-4 Expert workflow retains package cancellation and offload hooks. The expanded optional-runtime symbol contract passed clean-base locked install/activation/workload/rollback qualification. The deterministic 133-workflow catalog is graph-qualified/runtime-unqualified; the missing license file, post-training, and safety checker keep Auto and Gallery disabled. The required app-only aggregate preflight was 36.31 GB short, so the snapshot was not submitted and no older model was deleted; no media has been generated. |
| P6.52 Krea 2 Raw/Turbo standard source and admission-gate review | `e176f14` | Not required | Static gated source/license review only; task-scoped terms acceptance, commercial eligibility/legal approval, downstream terms/content-filter implementation, immutable AUP receipt, backend-owned bounds/runtime admission, app capacity, remote heavy-hardware output review, and physical macOS execution pending | Not required | Two exact gated official revisions, two Python-free 17-file / 35.68 GB Diffusers candidate partitions, five immutable safetensors weights per recipe, duplicate native-checkpoint exclusions, pinned package source receipts, distinct Raw/Turbo recipes, and estimate-only envelopes are sealed. Custom terms, mandatory content filtering, missing package safety checker/bounds, and two queue-aware app preflight deficits of about 53.2 GB keep the standard family outside every runtime/download/user-facing surface. Terms were not accepted, no app POST occurred, no older model was deleted, and no weight bytes or media were fetched. |
| P6.53 Stable Diffusion 3 standard source and admission-gate review | `e6061d9` | Not required | Static gated source/license review only; task-scoped terms acceptance, commercial license/legal approval, authenticated component review, backend-owned bounds/runtime admission, app capacity, remote heavy-hardware output review, and physical macOS execution pending | Not required | The exact gated official revision, Python-free 31.01 GB snapshot, six-file / 15.50 GB fp16 base inventory shared with the prior SD3 ControlNet review, three package-owned routes, source receipts, recipes, and estimate-only envelope are sealed. Noncommercial-only terms, inaccessible gated configs, missing safety checker/bounds, and a 21.41 GB queue-aware app preflight deficit keep the family outside every runtime/download/user-facing surface. Terms were not accepted, no app POST occurred, no older model was deleted, and no base weight bytes were fetched. |
| P6.54 Long-video continuation boundary handoff | `67010c7` | Not required | Synthetic graph/loop proof only; durable restart checkpoints, retained-asset stitching, mux/cancellation recovery, final 30-minute graph, remote execution, and physical macOS pending | Not required | The generic planner now binds the first opening anchor and the generic shot executor consumes the preceding loop segment only for explicitly marked continuation jobs. Missing carry and unknown strategies fail before inference; a two-iteration synthetic graph proves exact last-frame handoff. No model, media, or live inference was used. |
| P6.55 Long-video component recovery gate | `53e22f2` (revalidates `76bbafe`) | Not required | Synthetic loop interruption/resume and temporary real-file FFmpeg proof only; process-restart persistence, final 30-minute graph, remote execution, and physical macOS pending | Not required | A completed loop segment survives node-cache clearing and cancellation recovery without regeneration. Two temporary retained MP4s stitch deterministically to 14 frames / 1.75 seconds and retain those values after audio mux. No model or live inference was used, and all test media was temporary. |
| P6.56 Bounded 30-minute chunk planner | `88b5d33` | Not required | Deterministic planning proof only; durable restart recovery follows in P6.57, while the executable graph, remote execution, and physical macOS remain pending | Not required | Exact 1,800-second LTX planning produces 374 bounded continuation jobs at 16 FPS with explicit execution controls and a 512-job default ceiling. Oversized duration/job-count and unqualified long single-job FramePack plans fail before inference. No model or media was used. |
| P6.57 Durable long-video loop restart recovery | `d648417` | `19620f9` | Synthetic process-replacement and temporary retained-file proof only; executable 30-minute graph, remote execution, and physical macOS pending | Not required | Opt-in durable loops persist bounded managed video-asset metadata after each completed iteration, bind recovery to the exact workflow/input identity, and remove the checkpoint on graph success. A replacement server with a different task ID resumes after segment 1 and runs only segment 2; generic and in-memory loops remain non-durable. |
| P6.58 Executable 30-minute LTX qualification graph | `479d495` (`a9cf15c` schema-boundary fix) | Not required | Static graph/materializer and deterministic plan proof only; remote six-hour execution, output review/publication, and physical macOS pending | Not required | One reviewed API graph binds the exact LTX revision, 374-job continuation plan, durable retained-segment loop, and file-native join. The loopback-only helper binds the staged opening-image bytes to the recovery identity, verifies the exact app-cached model revision, and requires explicit consent before app submission. No graph was submitted and no inference or media generation occurred locally. |
| P6.59 Wan Animate 2 Modular contract closure | `d31d5b6` | `fc67a7f` | Contract-only; artifact admission, real-weight remote execution, output review, and physical macOS pending | Not required | Both pinned package exports are Expert-visible with exact generic character-animation contracts and distinct base/distilled denoise steps. The 33-class / 93-workflow snapshot matches the complete pinned exported-class set; complete backend and client gates pass. No repository, runnable mode, Auto/template/Gallery surface, weights, models, or media were added or removed. |
| P6.60 Wan Animate 2 artifact/source review | `2b86e63` | Not required | Static immutable metadata/package-source review only; immutable component descriptors, remote compiled flex-attention execution/output review, and physical macOS pending | Not required | Two public Python-free 31-file snapshots and their distinct 45,920,934,868-byte safetensors receipts are sealed. The standard class is absent at the pin and both Modular indexes retain null and mutable PR component revisions, so no runtime/download catalog or runnable mode was admitted and no weights or media were fetched. |
| P6.61 Bounded AuraFlow app download selection | `ed3982a` | Not required | Exact immutable app plan only; remote real-weight execution/output review and physical macOS remain pending | Not required | Plan and POST now derive one exact 18-file / 16,837,479,394-byte fp16 runnable selection from the capability, excluding four duplicate/default weight surfaces and the unrelated single-file/ComfyUI artifacts. The pre-existing full-repository app transfer remains untouched and no cache entry was deleted. |
| P6.62 Bounded Chroma app download selection | `0922243` | Not required | Exact immutable app plan only; remote real-weight execution/output review and physical macOS remain pending | Not required | Plan and POST now derive one exact 18-file / 27,493,360,428-byte runnable Diffusers selection, excluding the 17,800,038,288-byte duplicate native checkpoint and demo artifacts. The selected files are already complete in the preserved full cache, so no POST or deletion occurred. |
| P6.63 Bounded Allegro, Latte, and Mochi app download selections | `cb3448d` | Not required | Exact immutable app plans only; remote real-weight execution/output review and physical macOS remain pending | Not required | Exact 18/18/21-file runnable selections replace repository-wide planning and exclude Allegro's unsafe `.bin` duplicates, Latte's unsafe `.pt` plus unused decoder, and Mochi's 93.48 GB of duplicate/default partitions. Current queue-aware plans do not fit, so no POST or deletion occurred. |
| P6.64 Bounded Stable Audio and Stable Video app download selections | `8c96321` | Not required | Exact immutable app plans only; remote real-weight execution/output review and physical macOS remain pending | Not required | Exact 19-file / 5,348,079,831-byte Stable Audio and 12-file / 4,509,218,296-byte Stable Video runnable selections replace repository-wide planning. They exclude 24.14 GB of duplicate original, default/full-precision, and demo surfaces while retaining safetensors-only runtime components and rights receipts. Queue-aware plans do not fit beside current reservations, so no duplicate POST, interruption, or deletion occurred. |
| P6.65 Bounded AudioLDM2 and Shap-E app download selections | `56faa30` | Not required | Exact immutable app plans only; remote real-weight execution/output review and physical macOS remain pending | Not required | Exact 28-file / 4,480,959,446-byte AudioLDM2 and 14-file / 1,332,951,857-byte Shap-E safe-component selections replace repository-wide planning and exclude 8.04 GB of legacy pickle and unsafe duplicate surfaces. Both selections are already complete in the preserved full caches, so no POST or deletion occurred. |
| P6.66 Bounded Marigold app download and loader serialization | `c841203` | Not required | Exact immutable app plan and prior local tiny smoke only; remote quality/output review and physical macOS remain pending | Not required | One exact 14-file / 5,161,610,352-byte float32 safetensors selection replaces repository-wide planning, excludes 10.32 GB of legacy pickle and duplicate fp16 surfaces, and makes the loader's safe-serialization requirement explicit. The selection is already complete; no POST or deletion occurred, and older extra/unfinished blobs remain preserved. |
| P6.67 Bounded Whisper Tiny app download selection | `4783400` | Not required | Exact immutable app plan and prior local in-memory ASR smoke only; remote semantic/output review and physical macOS remain pending | Not required | One exact 13-file / 155,455,649-byte safetensors/processor selection excludes 453,397,578 bytes of duplicate PyTorch, Flax, and TensorFlow weights. Only 21,225 metadata bytes remained; a fresh fitting plan preceded the app-only completion request, with no deletion. |
| P6.68 Bounded unconditional-image app downloads and loader serialization | `42fa625` | Not required | Exact immutable app plans and prior local tiny smokes only; remote quality/output review and physical macOS remain pending | Not required | Exact six-file safetensors selections for shared DDPM/DDIM CIFAR-10 and ImageNet64 consistency routes exclude 1.33 GB of legacy pickle plus repository code/demo surfaces and make all three loader serialization requirements explicit. Only 15,704 receipt bytes remained; fresh fitting plans preceded two concurrent app-only repairs, with no deletion. |
| P6.69 Bounded LCM DreamShaper app download and loader serialization | `08c34f4` | Not required | Exact immutable app plan and prior local tiny smoke only; remote quality/output review and physical macOS remain pending | Not required | One exact 17-file / 5,482,979,343-byte safetensors component selection excludes 7,710,367,026 bytes of duplicate single-file, ONNX, repository-code, and demo surfaces and makes the generic loader's safe-serialization requirement explicit. Only 5,070 receipt bytes remained; a fresh fitting plan preceded the app-only completion request, with no deletion. |
| P6.70 Bounded Sana 0.6B fp16 app download selection | `982c1a3` | Not required | Exact immutable app plan only; remote real-weight execution/output review and physical macOS remain pending | Not required | One exact 17-file / 7,700,017,758-byte fp16 Diffusers selection matches the loader variant and excludes 8,844,837,635 logical bytes of default/full-precision aliases. The selected files are already complete in the preserved cache, so no POST or deletion occurred. |
| P6.71 Bounded FLUX.2 Klein component download and loader serialization | `80f7369` | Not required | Exact immutable app plan and prior local live smokes only; remote output review and physical macOS remain pending | Not required | One exact 21-file / 15,980,152,900-byte component selection excludes the 7,751,105,712-byte native duplicate and demo images while making both generic Klein loaders safetensors-only. The selected files are already complete in the preserved cache, so no POST or deletion occurred. |
| P6.72 Bounded primary FLUX.1 component downloads and loader serialization | `274b158` | Not required | Exact immutable app plans only; remote real-weight execution/output review and physical macOS remain pending | Not required | Exact 25/26/26-file component selections for schnell, dev, and Krea total 101,218,748,600 bytes and exclude 72,409,924,695 logical bytes of native checkpoints, root autoencoders, and demos while making all three shared loaders safetensors-only. Every selected file is already complete in the preserved caches, so no POST or deletion occurred. |
| P6.73 Bounded conditioned FLUX.1 component downloads and loader serialization | `d39bfe2` | Not required | Exact immutable app plans only; remote real-weight execution/output review and physical macOS remain pending | Not required | Exact 28/28/26/26-file component selections for Depth, Canny, Fill, and Kontext total 155,032,793,063 bytes and exclude 96,561,961,318 logical bytes of native checkpoints, root autoencoders, and demo media while making all four loader paths safetensors-only. Every selected file is already complete in the preserved caches, so no POST or deletion occurred. |
| P6.74 Bounded shared SD1.5 image/video download union and image-loader serialization | `896a723` | Not required | Exact immutable app plan plus prior local image/video smokes only; remote output review and physical macOS remain pending | Not required | One exact 21-file / 8,223,292,159-byte union covers the float32 image/PAG routes and fp16 AnimateDiff/AnimateLCM routes, excludes 39,036,647,490 bytes of pickle, non-EMA, single-file, and YAML surfaces, and makes all five image adapters safetensors-only. A fresh fitting plan preceded the app-only 2,740,639,959-byte fp16 completion request, with no deletion. |
| P6.75 Bounded Z-Image component download and loader serialization | `8f96945` | Not required | Exact immutable app plan and prior local live smokes only; remote output review and physical macOS remain pending | Not required | One exact 21-file / 32,848,321,404-byte component selection excludes 51,345,993 bytes of gallery/PDF assets and makes all three generic Z-Image adapters safetensors-only. Every selected file is already complete in the preserved cache, so no POST or deletion occurred. |
| P6.76 Bounded PixArt Sigma component download selection | `a0fe20c` | Not required | Exact immutable app plan only; remote real-weight execution/output review and physical macOS remain pending | Not required | One exact 15-file / 21,828,231,839-byte component selection excludes 4,258,550 bytes of documentation images and matches the existing safetensors-only loader. Every selected file is already complete; the active aggregate queue temporarily exceeded the safety envelope by 374,414,385 bytes, so no POST, interruption, or deletion occurred. |
| P6.77 Bounded admitted Wan video component downloads and loader serialization | `e3fc376` | Not required | Exact immutable app plans plus prior local Wan smokes only; remote output review and physical macOS remain pending | Not required | Exact 21/19/43/22-file component selections for Wan 2.1 T2V/edit, VACE, Wan 2.2 I2V, and TI2V total 208,370,069,191 bytes, exclude 15,892,213 bytes of documentation/example media, and make every corresponding VAE/pipeline load explicitly safetensors-only. Every selected file is already complete; the active aggregate queue temporarily exceeded the safety envelope by 2,661,386,301 bytes, so no POST, interruption, or deletion occurred. |
| P6.78 Bounded JoyAI Image Edit app download selections | `c0c5944` | Not required | Exact immutable app plans only; remote real-weight execution/output review and physical macOS remain pending | Not required | Exact 38/29-file component selections total 100,686,169,296 bytes and exclude 16,592,417 bytes of repository code plus test/example media while retaining both existing safetensors-only loader contracts. Edit is complete; the pre-existing repository-wide Plus app job remains active and differently scoped, so no new POST, join, interruption, or deletion occurred. |
| P6.79 Bounded ACE-Step component download and loader serialization | `96b71f9` | Not required | Exact immutable app plan only; remote real-weight execution/output review and physical macOS remain pending | Not required | One exact 21-file / 11,101,507,113-byte component selection excludes the 3,841,215-byte raw `silence_latent.pt` debug artifact; the runtime silence latent is already baked into the condition-encoder safetensors. The ACE-Step loader now explicitly requires safe serialization. The selected files are already complete; the active aggregate queue plus reserve exceeded free space by 2,662,033,371 bytes, so no POST, interruption, or deletion occurred. |
| P6.80 Bounded Sana Sprint app download selection | `da6a631` | Not required | Exact immutable app plan only; remote real-weight execution/output review and physical macOS remain pending | Not required | One exact 17-file / 7,703,536,042-byte selection seals the complete reviewed bfloat16 safetensors snapshot, including its license and bundled tokenizer terms, so future repository additions cannot silently expand the app download. Every selected file is already complete; the active aggregate queue plus reserve exceeded free space by 2,662,840,236 bytes, so no POST, interruption, or deletion occurred. |
| P6.81 Bounded DreamLite app download selections | `bcad80a` | Not required | Exact immutable app plans only; remote real-weight execution/output review and physical macOS remain pending | Not required | One shared exact 27-file component inventory seals the Base and Mobile snapshots at 10,143,788,478 logical bytes, including the separately consumed processor and tokenizer trees and the CC-BY-NC model card, with safetensors-only weights. Base was already complete. A fresh exact app plan on 2026-08-14 now reports all 27 Mobile files / 5,071,894,300 selected bytes complete at immutable revision `6695c3f4be230f0493fa5dbf78be3bc4d3bb2ab4` after the app-owned transfer finished. No duplicate POST, interruption, or deletion occurred, and all older model caches remain preserved. |
| P6.82 Bounded CogView4, ERNIE Image Turbo, and GLM-Image downloads | `9345725` | Not required | Exact immutable app plans only; remote real-weight execution/output review and physical macOS remain pending | Not required | Exact 21/24/27-file component selections seal 98,566,385,977 logical bytes across the three safetensors-only pipelines. Each manifest retains every model-index component and its applicable license/model-card evidence, including ERNIE's distinct prompt-enhancer tokenizers and GLM's processor plus vision-language encoder. All selected files are complete; the latest active aggregate queue plus reserve exceeded free space by 2,730,885,389 bytes, so no POST, interruption, or deletion occurred. |
| P6.83 Bounded Qwen Image family downloads and loader serialization | `54991a3` | Not required | Exact immutable app plans only; remote real-weight execution/output review and physical macOS remain pending | Not required | Exact 30/39/35/33-file component selections seal 230,866,011,411 logical bytes for Qwen Image 2512, Edit, Edit 2511, and Layered. Each manifest preserves its distinct processor/tokenizer and transformer shard topology, while all six standard 2512/Edit adapters now explicitly require safetensors; the Modular loader already did. Every selected file is complete; the active aggregate queue plus reserve exceeded free space by 2,664,392,168 bytes, so no POST, interruption, or deletion occurred. |
| P6.84 Bounded FLUX.1 Redux prior download and loader serialization | `1090d3e` | Not required | Exact immutable app plan only; remote real-weight execution/output review and physical macOS remain pending | Not required | One exact 9-file / 985,539,827-byte prior selection excludes 129,528,462 bytes of duplicate single-file weights and demo media while retaining both actual prior weight components, preprocess configuration, license, and model index. Both Redux and its fixed FLUX.1-dev base now load with explicit safe serialization. The selection is already complete; the active aggregate plan fit by only 73,247,222 bytes, so no redundant POST, interruption, or deletion occurred. |
| P6.85 Bounded CogVideoX-2B app download selection | `4b54458` | Not required | Exact immutable app plan only; remote real-weight execution/output review and physical macOS remain pending | Not required | One exact 17-file / 13,775,557,177-byte component selection retains the complete reviewed Apache-2.0 safetensors runtime and excludes 15,561 bytes of `.gitignore` and duplicate-language documentation. The existing loader already requires safe serialization, VAE tiling, and model CPU offload. No bytes are cached yet; the fresh aggregate app plan was short by 13,769,969,346 bytes, so no POST, interruption, or deletion occurred. |
| P6.86 Bounded SANA-Video text/image app download selection | `15bd08f` | Not required | Exact immutable app plan only; remote real-weight execution/output review and physical macOS remain pending | Not required | One exact 20-file / 14,002,562,288-byte component selection seals the complete reviewed Apache-2.0 safetensors snapshot shared by the text-to-video and image-to-video routes. The existing loaders retain their BF16 text/transformer, FP32 Wan VAE, safe serialization, mandatory VAE tiling, and sequential CPU offload contracts. No bytes are cached yet; the fresh aggregate app plan was short by 13,930,566,291 bytes, so no POST, interruption, or deletion occurred. |
| P6.87 Bounded LongCat AudioDiT app download selection | `ddb37ec` | Not required | Exact immutable app plan only; remote real-weight execution/output review, snapshot license declaration clarification, and physical macOS remain pending | Not required | One exact 13-file / 5,701,278,543-byte component selection seals the complete reviewed safetensors conversion used by the native 24 kHz audio route. The existing loader already requires safe serialization and sequential CPU offload; the model card retains the recorded upstream MIT-aligned provenance while the snapshot metadata itself does not declare a license. A fresh exact app plan on 2026-08-14 reports zero remaining bytes after the pre-existing app-owned transfer completed. No duplicate POST, interruption, or deletion occurred, and the preserved cache remains available for preview regression testing. |
| P6.88 Bounded SDXL Base, Turbo, and InstructPix2Pix downloads | `18248e8` | Not required | Exact immutable app plans plus prior source/contract qualification only; remote real-weight execution/output review and physical macOS remain pending | Not required | Exact 21/21/20-file component selections seal 26,125,893,532 logical bytes across SDXL Base, Turbo, and InstructPix2Pix. Base and Turbo retain only their fp16 Diffusers variants; InstructPix2Pix retains its component-only safetensors tree and excludes 600 validation/demo files. All eight base-backed generic adapters now explicitly request the fp16 safetensors variant, while InstructPix2Pix remains safetensors-only without a variant suffix. The first aggregate plans did not fit; after active reservations fell, a fresh fitting plan preceded the exact 6,941,211,875-byte Base app request, which remains active rather than complete. Turbo and InstructPix2Pix were not submitted, and no older cache entry was interrupted or deleted. |
| P6.89 Bounded CogView3 Plus, Kandinsky 3, and Hunyuan-DiT downloads | `43480cb` | Not required | Exact immutable app plans plus prior source/contract qualification only; remote real-weight execution/output review and physical macOS remain pending | Not required | Exact 20/19/20-file component selections seal 68,379,122,391 logical bytes across CogView3 Plus, Kandinsky 3, and the shared Hunyuan-DiT v1.2 distilled base. They preserve each model-index runtime tree and its bfloat16, fp16-variant, or unsuffixed safetensors topology while excluding deployment-only configuration, duplicate-language documentation, and 11 Kandinsky demo images. Fresh exact app plans on 2026-08-14 report all 20 CogView3 Plus files / 25,560,117,047 selected bytes complete at immutable revision `5d70e40732ac0efac98524c51a7fa9c82707f1e5` and all 19 Kandinsky 3 files / 28,391,690,623 selected bytes complete at immutable revision `bf79e6c219da8a94abb50235fdc4567eb8fb4632` after their pre-existing app-owned transfers finished. The Hunyuan-DiT base was not newly submitted, and the separate Hunyuan Canny component remains app-queued rather than complete. No duplicate POST, interruption, or deletion occurred, and all older model caches remain preserved. |
| P6.90 Bounded Lumina Next and OmniGen downloads | `1d1784c` | Not required | Exact immutable app plans plus prior source/contract qualification only; remote real-weight execution/output review and physical macOS remain pending | Not required | Exact 16/11-file safetensors component selections seal 16,967,647,146 logical bytes across Lumina Next and OmniGen, including every immutable snapshot file used by their package-owned pipelines. The long-prompt LTX image-to-video capability now also publishes the same existing 22-file bounded selection as the ordinary LTX capability instead of appearing unbounded. Fresh exact app plans on 2026-08-14 report both selections complete at immutable revisions `0ee5ec90043acf5cb41fe96274af36eb7fad8d95` and `016e2f61d12a98303f6bbdf122687694d7984268`; both cache entries are installed, complete, and repair-free after their app-owned transfers. No duplicate POST, interruption, or deletion occurred. |
| P6.91 Bounded FramePack three-repository assembly | `76697aa` | `aeb0055` | Exact immutable app plans plus prior source/contract qualification only; remote real-weight execution/output review, repository-license clarification, and physical macOS remain pending | Not required | Exact 7/21/5-file selections seal 42,866,796,087 logical bytes across the FramePack transformer, HunyuanVideo scheduler/encoder/tokenizer/VAE base components, and FLUX Redux SigLIP processor/vision encoder. The base selection omits its unused 25.64 GB transformer tree, the vision selection omits the unused image embedder, every weight-bearing loader now requires safetensors, Studio setup exposes both auxiliary repositories, and the app resolves reviewed dependency selections even when an older client omits explicit file arguments. Fresh queue-aware plans were short by 33,765,762,846, 24,278,214,885, and 8,873,353,899 bytes respectively, so no FramePack POST, interruption, or deletion occurred. |
| P6.92 Bounded remaining admitted LongCat and Wan downloads | `e57b677` | Not required | Exact immutable app plans plus prior source/contract qualification only; remote real-weight execution/output review, recorded license clarifications, and physical macOS remain pending | Not required | Exact 31/32/43/41-file component selections seal 274,936,785,599 logical bytes across LongCat Image, LongCat Image Edit, Wan 2.2 T2V A14B, and Wan 2.1 first/last-frame video. The LongCat selections retain both separately loaded processor/tokenizer trees while excluding root deployment configuration and demo assets; the Wan selections retain every safetensors model-index component while excluding documentation media. Pinned remote path parity and 193 focused tests pass. Fresh exact app plans on 2026-08-14 now report all 31 LongCat Image files / 29,316,541,013 selected bytes complete at immutable revision `d2ea50b79a930074c37b9b97ce45e3b2ea8cf4d8` and all 32 LongCat Image Edit files / 29,316,540,813 selected bytes complete at immutable revision `7b54ef423aa7854be7861600024be5c56ab7875a` after their pre-existing app-owned transfers finished. Both cache entries are installed, complete, and repair-free; both Wan selections remain pending. No duplicate POST, interruption, or deletion occurred, and all older model caches remain preserved. Every admitted Studio primary model repository now has a bounded manifest except LTX-2's gated artifact and Wan Animate's unresolved component-revision contract, which remain deliberately fail-closed. |
| P6.93 Bounded admitted auxiliary downloads | `cc748d0` | Not required | Exact immutable app plans plus prior source/contract qualification only; current-worker activation, remote real-weight execution/output review, and physical macOS remain pending | Not required | Exact 4/4/4/4/4/4/5-file selections seal 13,476,063,409 logical bytes across Qwen Image ControlNet Union, Hunyuan-DiT v1.2 Canny ControlNet, SDXL T2I-Adapter Canny, SDXL ControlNet Canny, AnimateDiff MotionAdapter, SD1.5 ControlNet Canny, and AnimateLCM MotionAdapter plus LoRA. The selections exclude repository Python, demo media, legacy `.bin`/`.ckpt` weights, and unused full-precision duplicates while retaining the precise safe variant each existing loader consumes. Every admitted dependency repository now resolves a non-empty app selection; the FLUX Redux base requirement reuses its already-bounded primary FLUX.1-dev selection. Fresh exact app plans confirm all seven immutable sizes and report the Qwen selection's 3,536,036,150 bytes complete; the other six pre-existing differently scoped app jobs remain untouched. The live worker predates this commit, so automatic bounded selection is source/test evidence for the later safe restart rather than a claim about current transfers. The affected 277-test matrix passes with 645 subtests and five expected skips; the complete backend gate passes 1,690 tests, 3,438 subtests, and 40 expected skips together with Ruff, 66-package compatibility, shell, preflight, and diff checks. No new POST, interruption, deletion, generation, output, or physical macOS qualification occurred. |
| P6.94 Reconnect-safe app download visibility | `33d1754` | `b0c106b` | Source/unit and exact bundled-client evidence only; current-worker activation, active-transfer completion, remote generation, and physical macOS remain pending | Not required | The app now retains each active model transfer's latest bounded progress receipt, exposes a read-only schema-v1 `/hf_download/status` response without selected file paths, and includes the same authoritative active list in every WebSocket welcome. A reconnect replaces stale active client entries while preserving terminal history, so refreshing the browser no longer loses app-owned queue visibility or invents completion. The exact client build is mirrored into the backend. Focused backend coverage passes 41 tests and 95 subtests; the complete backend gate passes 1,691 tests, 3,438 subtests, and 40 expected skips with Ruff, compile, shell, and 66-package compatibility green. The complete client check passes, including the new reconnect regression, and the production bundle remains within its 533,504-byte ceiling at 533,425 gzip bytes. The Playwright scenario was not executed because this Linux checkout has no installed browser binary; no browser package was downloaded merely for this slice. The old live worker was not restarted while app transfers remain active, and no model/cache byte was downloaded outside the app, deleted, generated, or treated as macOS evidence. |
| P6.95 Exact join-aware app download planning | `a0a161a` | Not required | Source/unit evidence only; current-worker activation, active-transfer completion, remote generation, and physical macOS remain pending | Not required | A queue-aware plan for the same repository, immutable revision, and exact file selection as an active app task now excludes that task's existing reservation instead of counting it twice. The response identifies the already-queued task and reports a join as fitting without weakening the atomic reservation check or the HTTP 409 boundary for a conflicting revision/selection. Focused model/Gallery coverage passes 28 tests and 66 subtests; the complete backend gate passes 1,692 tests, 3,438 subtests, and 40 expected skips with Ruff and compile green. The old live worker still reports the pre-fix planning shape and was not restarted while transfers remain active. No task was duplicated, cancelled, interrupted, deleted, generated, or treated as macOS evidence. |
| P6.96 Qualification/download mutual exclusion | Not required | `3413455` | Source/unit and production-build evidence only; current-worker activation, active-transfer completion, remote generation, and physical macOS remain pending | Not required | A real release-qualification campaign now requires the app's bounded schema-v1 download status to be idle before startup and rechecks it before every model-family group; the Gallery runner performs the same fail-closed check before startup and immediately before every selected template. Dry runs can request the check explicitly with `--check-download-idle`. Focused campaign/runner coverage passes 55 tests, and the complete client check passes formatting, lint, typechecking, all unit suites, production build, and the 533,425 / 533,504-byte gzip bundle budget. A generation-free dry run against the old live worker still selects 76 jobs in six groups and reports 76/76 model-ready jobs, but correctly blocks 38 jobs on 50 absent Gallery input assets totaling 33,867,388 bytes; the new status route was not called because that worker predates it. The worker was not restarted while app transfers remain active, and no graph was submitted, no model/cache byte was deleted or downloaded outside the app, and no Linux result was treated as physical macOS evidence. |
| P6.97 Catalog-wide app transfer coverage audit | Not required | Not required | Exact immutable read-only app plans and live-caller/monitor inventory only; transfer completion, remote generation, and physical macOS remain pending | Not required | All 66 admitted capabilities collapse to 47 unique primary repository/revision/file selections: 23 are exact-complete and 24 retain bytes. Twenty-three pending primaries were already represented by a long-lived app request or the bounded admission monitor; the sole omission was the exact 38-file / 51,656,728,957-byte `NucleusAI/Nucleus-Image@5e963db4fd0a65c7e4faf53ca2d4eca567c4dcfa` selection. A dedicated app-only monitor now performs two fresh queue-aware plans before any future submission. Its first plan reported 178,564,902,912 free bytes, 116,682,001,651 queued reservation bytes, and the 68,719,476,736-byte reserve, so it correctly did not fit and no POST occurred. All eight unique auxiliary model requirements are likewise represented by existing live app callers or monitors. The audit downloaded no model byte directly, deleted no cache, submitted no graph, and made no output or macOS claim. |
| P6 remaining | Pending | Pending | Remote pending | Pending | LTX-2.5 gated artifact/live qualification, other heavy families, and long-form workflow qualification remain open as independent segments. Kandinsky5 Video artifact/source evaluation is complete in backend `08e2550`, with corrected recipe evidence and remote execution still pending. |
