# Hugging Face Integration Roadmap

This document is the implementation and completion tracker for closing MoDiff's
official Diffusers, Modular Diffusers, and approved Hugging Face speech-runtime
gaps. It is a durable product roadmap rather than a claim that every upstream
pipeline is already runnable.

The reviewed Diffusers installation remains pinned to commit
`13a7bee4878d62fccc8d25f97e480e68de96fa03`. The latest-upstream check on
2026-08-12 found [Diffusers v0.39.0](https://github.com/huggingface/diffusers/releases/tag/v0.39.0)
as the latest tagged release (release commit
`a3608b512ed7248499a44c61d954965ed9bdae4d`) and
`175fe6b2419a01db9c2ceabd01ec37d2c0305fc2` as the latest `main` commit. The
comparison inventory now uses the reviewed `main` snapshot
`175fe6b2419a01db9c2ceabd01ec37d2c0305fc2`. Re-run the inventory before
changing the Diffusers pin or marking a gap complete.

### Upstream delta reviewed 2026-08-12

The five commits after the 2026-08-09 snapshot contain one model/workflow
change: upstream commit
[`7564fb0`](https://github.com/huggingface/diffusers/commit/7564fb0) adds
LTX-2.5. The other four commits are an import guard, device deduction, LoRA
bookkeeping, and NVIDIA Spark installation documentation; they add no pipeline
family. LTX-2.5 reuses the standard `LTX2Pipeline` family rather than adding a
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
      hosted architecture, retains the explicit prospective dependency diff,
      requires ready preflight, runs the consented qualifier, and uploads bounded
      evidence for review. It has not run and makes no macOS success claim.
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
        removed; ports 8088/8089 were free afterward.
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

- [ ] **P3.1 Generic unconditional image generation**
  - Add an unconditional task mode/adapter, not DDPM-specific nodes.
  - Integrate `DDPMPipeline`, `DDIMPipeline`, and
    `ConsistencyModelPipeline` in separate exact-pair entries.
  - Local smoke: tiny resolution and bounded steps; hard timeout 40 minutes.
- [ ] **P3.2 Small latent image workflows**
  - Stable Diffusion 1.x/2.x text-to-image, img2img, and inpaint.
  - LCM 1-4 step workflows and PAG using compatible base weights.
  - Local smoke: at most 512px and the minimum meaningful step count.
- [ ] **P3.3 Generic perception output**
  - Add prediction-map output semantics and integrate Marigold depth first.
  - Add normals, intrinsics, and uncertainty only after the shared output
    contract is stable.
- [x] **P3.4 Adopt the official Hugging Face library boundary in repository policy**
  - Update backend and client `AGENTS.md`, contributor guidance, Hugging Face
    standards, dependency/runtime contracts, and the former Diffusers-only
    boundary tests.
  - Permit reviewed official Hugging Face libraries while preserving the single
    MoDiff graph executor, local execution, immutable sources, no implicit
    remote code, generic contracts, rollback, and proof requirements.
  - This is a documentation/contract commit and generates no media.
  - Status 2026-08-07: implemented and validated in both working trees; paired
    commit references are pending. The current direct Transformers and PEFT
    dependencies remain an explicitly documented migration gap for P0.5.
- [ ] **P3.5 Hugging Face Transformers speech-to-text implementation**
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

### Phase 3 test and asset gate

- [ ] Static signature and artifact-policy tests pass for every exact model/mode.
- [ ] Tiny/mocked output normalization tests pass.
- [ ] Paired client forms, graph bridges, readiness, errors, and browser flows
  pass.
- [ ] Each local smoke completes below 40 minutes or is moved to remote without
  a local retry.
- [ ] Remote media and ASR fixtures pass rights review and Dataset verification.
- [ ] Only exact live-qualified recipes may enter Auto.

## Phase 4 — Medium image, audio, and 3D integrations

Priority: after the small-model contracts are stable. Hardware: remote by
default. Assets: remote Dataset only.

### Committable segments

- [ ] **P4.1 Control adapters:** SD1.5 ControlNet and T2I Adapter with pinned
  preprocessors and auxiliary models.
- [ ] **P4.2 SDXL expansion:** Turbo first, then the reviewed text, image,
  inpaint, instruct, ControlNet, adapter, PAG, and related combinations.
- [ ] **P4.3 Moderate image families:** DreamLite, Sana/Sana Sprint, and other
  candidates admitted by the per-model checklist.
- [ ] **P4.4 Audio generation:** LongCat AudioDiT, Stable Audio quality recipes,
  and AudioLDM2 general audio.
- [ ] **P4.5 Diffusers text-to-speech:** AudioLDM2 TTS with a generic speech
  synthesis task contract. Require a reviewed safetensors artifact or an
  explicit documented unsafe-deserialization exception before execution.
- [ ] **P4.6 Generic 3D artifacts:** Shap-E rendered output first; mesh/PLY/OBJ/GLB
  only after a safe artifact/export contract exists.

### Phase 4 test and asset gate

- [ ] Unit and tiny-fixture tests cover adapters, outputs, and cleanup.
- [ ] Backend/client integrated gates pass for each independent segment.
- [ ] No Phase 4 live model is required to run locally.
- [ ] Remote receipts include peak memory, runtime, dependency/model revisions,
  graph hash, media checks, and cleanup result.
- [ ] Gallery activation follows rights and anonymous byte verification.

## Phase 5 — Short video qualification

Priority: after image/audio contracts. Hardware and assets: remote only.

### Committable segments

- [ ] Qualify the existing Wan, LTX/LTX2, and Hunyuan FramePack graph paths from
  Phase 2 using minimal short outputs.
- [ ] Add Stable Video Diffusion using documented offload and decode chunking.
- [ ] Add AnimateDiff/AnimateLCM with separately pinned base model,
  `MotionAdapter`, scheduler rules, and optional LoRA.
- [ ] Evaluate Motif Video, CogVideoX-2B, and similar smaller candidates one at a
  time after artifact-size and RAM review.

### Phase 5 test and asset gate

- [ ] Static and mocked tests cover frame count, dimensions, conditioning,
  scheduler/adapter compatibility, output normalization, and cleanup.
- [ ] Remote smoke uses the minimum supported 8-25 frames and bounded steps.
- [ ] Representative quality proof is limited to approximately 2-4 seconds.
- [ ] Non-black frames, finite tensors, duration/frame rate, and decode/mux
  integrity are checked without cross-hardware pixel hashes.
- [ ] Auto remains disabled until the exact short-video recipe has live proof.

## Phase 6 — Pin update, heavy models, and long-form workflows

Priority: last. Hardware and assets: dedicated remote qualification only.

### Committable segments

- [ ] Review all commits between the current and proposed Diffusers pins; update
  the executable dependency, compatibility test, and upstream contract tests in
  one isolated change.
- [ ] Add the post-pin Krea2 and Krea2 Turbo Modular classes.
- [ ] Add `MiniMaxH3ModularPipeline` only through generic joint video+audio
  specifications for its distinct `t2va`, `fl2va`, and `ref2va` workflows.
  Validate the `transformer/` versus `transformer_ref/` partition receipt,
  Qwen3-VL conditioning, separate video/audio scheduler state, reference-media
  bounds, immutable artifact revision, and remote-only resource envelope before
  exposing any mode.
- [ ] Add the post-pin `LTX2ModularPipeline` and `LTX25ModularPipeline`, then
  separately qualify LTX-2.5 distilled single-stage, full/SFT plus stage-2 LoRA,
  and distilled two-stage recipes. Bind the exact sigma schedules, latent
  upsampler, duration head, Gemma-4 prompt enhancer, diffusion decoder/NATTEN
  path, and audio/video output handoff; prompt enhancement must remain an
  explicit execution action and may not download during discovery or planning.
- [ ] Evaluate HunyuanVideo 1.5, Helios/Pyramid, Wan 14B/22 Modular, full LTX/LTX2,
  EasyAnimate, SkyReels, Cosmos/Cosmos3, Kandinsky5 Video, and other heavy video
  families.
- [ ] Evaluate large image/cascaded families and DiffusionGemma only on hardware
  with sufficient RAM, VRAM, and disk.
- [ ] Keep LLaDA2 blocked unless its remote-code requirement receives an explicit
  immutable-code security review.
- [ ] Build the 30-minute video workflow only after chunk generation, checkpoint
  resume, deterministic stitching, audio mux, cancellation, and recovery pass
  independently.

### Phase 6 test and asset gate

- [ ] No Phase 6 live run occurs on the current development machine.
- [ ] Heavy integrations can merge contract-only while clearly Expert-only and
  `qualification_pending`.
- [ ] Long-form component tests use synthetic/tiny segments.
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

The current inventory contains 20 classes: 15 present at the MoDiff pin and 5
that require a pin update. The latter group includes the two LTX2 exports added
after the previous roadmap snapshot.

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

Require a pin update:

- [ ] `Krea2ModularPipeline`
- [ ] `Krea2TurboModularPipeline`
- [ ] `MiniMaxH3ModularPipeline`
- [ ] `LTX2ModularPipeline`
- [ ] `LTX25ModularPipeline`

## Appendix B — Missing standard pipeline families

This is a family inventory, not a requirement to create one node per family.

### Audio

- [ ] `audioldm2`
- [ ] `longcat_audio_dit`

### Text diffusion

- [ ] `diffusion_gemma`
- [ ] `llada2`

### 3D and perception

- [ ] `shap_e`
- [ ] `marigold`
- [ ] `visualcloze`

### Video

- [ ] `allegro`
- [ ] `animatediff`
- [ ] `anyflow`
- [ ] `chronoedit`
- [ ] `cogvideo`
- [ ] `consisid`
- [ ] `cosmos`
- [ ] `easyanimate`
- [ ] `helios`
- [ ] `hunyuan_video1_5`
- [ ] `kandinsky5`
- [ ] `latte`
- [ ] `lucy`
- [ ] `mochi`
- [ ] `motif_video`
- [ ] `sana_video`
- [ ] `skyreels_v2`
- [ ] `stable_video_diffusion`

### Image, unconditional, and generic

- [ ] `aura_flow`
- [ ] `bria`
- [ ] `bria_fibo`
- [ ] `chroma`
- [ ] `cogview3`
- [ ] `cogview4`
- [ ] `consistency_models`
- [ ] `controlnet`
- [ ] `controlnet_hunyuandit`
- [ ] `controlnet_sd3`
- [ ] `ddim`
- [ ] `ddpm`
- [ ] `deepfloyd_if`
- [ ] `dit`
- [ ] `dreamlite`
- [ ] `ernie_image`
- [ ] `glm_image`
- [ ] `hidream_image`
- [ ] `hunyuan_image`
- [ ] `hunyuandit`
- [ ] `ideogram4`
- [ ] `joyimage`
- [ ] `kandinsky`
- [ ] `kandinsky2_2`
- [ ] `kandinsky3`
- [ ] `kolors`
- [ ] `krea2`
- [ ] `latent_consistency_models`
- [ ] `latent_diffusion`
- [ ] `ledits_pp`
- [ ] `longcat_image`
- [ ] `lumina`
- [ ] `lumina2`
- [ ] `nucleusmoe_image`
- [ ] `omnigen`
- [ ] `ovis_image`
- [ ] `pag`
- [ ] `pixart_alpha`
- [ ] `prx`
- [ ] `sana`
- [ ] `stable_cascade`
- [ ] `stable_diffusion`
- [ ] `stable_diffusion_3`
- [ ] `t2i_adapter`

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
| P0.5 | `4073711` (qualifier), `655baa6` (platform cutover), corrected by `1e95362` | `16046ab` (target-aware Setup status) | Windows x86-64 guarded live-model proof; Linux x86-64 clean-base/no-weight, supervised lifecycle, and production-cutover proof; macOS pending and base-delivered | Not required | Complete for qualified x86 targets: the exact six-target profile/delivery table enables explicit first-use install/activation only on Linux and Windows x86-64. Direct base dependencies remain only on macOS/ARM targets. A committed clean Linux CPU base contained 61 packages and none of the ten staged distributions; the exact overlay installed, validated, activated, passed the finite CLIP+LoRA child, rolled back, and restored a fresh clean base. A fresh worker exposed actionable status and rejected required execution with `optional_runtime_missing` before queueing. macOS and ARM rows remain explicitly non-actionable/base-delivered pending their own qualifier evidence. |
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
| P2.5 | Pending | Pending | Pending | Pending | Not started |
| P3.4 | Pending | Pending | Not required | Not required | Policy implementation and gates complete; paired commits pending |
| P3.1-P3.3, P3.5 | Pending; add one row per slice | Pending; add one row per slice | Pending | Pending | Not started |
| P4.1-P4.6 | Pending | Pending | Remote pending | Pending | Not started |
| P5 | Pending | Pending | Remote pending | Pending | Not started |
| P6 | Pending | Pending | Remote pending | Pending | Not started |
