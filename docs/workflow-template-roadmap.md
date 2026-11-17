# Workflow and template suite roadmap

Build a complete, usable **local workflow suite by task**, with reusable generic
stages and accepted examples. Keep the existing 89-case Gallery campaign as its
own completion milestone. Additional models, workflow declarations, comparison
images and catalogue variants do not silently change that denominator.

This roadmap combines the current MoDiff contracts with the reviewed official
ComfyUI catalogue and current official FLUX 3/MiniMax image sources. It is a
delivery plan, not a declaration that every listed integration already works.
Execution checkpoints, private media and per-file approvals stay in the campaign
ledger outside public documentation.

## Scope and catalogue coverage

The reviewed [ComfyUI index](https://github.com/Comfy-Org/workflow_templates/blob/a8bc07a76478dfada740fa8fa49ac87be1bff519/templates/index.json)
has 760 named entries across Image, Image Tools, Video, Video Tools, Audio, 3D,
Vector, LLM, Node Basics, Product Ads, Character/Fashion and Brand Design. The
comparison inspected the index and eight representative nested workflow files
as data; it did not execute all 760 graphs. Hosted APIs, model variants and
community runtimes prevent a meaningful one-to-one comparison with 89 examples.

Cover all twelve categories explicitly. Image/video/audio/3D/understanding and
utility families below provide the local scope. Product Ads, Character/Fashion
and Brand Design become useful compositions of those same families. Hosted API
and hosted-vector entries remain recorded exclusions, rather than hidden gaps.

Use these implementation states independently of quality/publication:

- **Existing:** a relevant implementation is present; the stated variants still
  need exact recipe and live-output checks.
- **Extend:** work belongs in an existing generic operation/backend adapter.
- **Review:** runtime provenance, checkpoint, licence, native task or resource
  admission is not established.
- **Upstream dependency:** a compatible trained model and official-library
  execution path are required before MoDiff can expose a working feature.
- **Excluded:** outside the current local-runtime product boundary.

## FLUX 3: why native masks and adapters are absent

The [reviewed current official FLUX 3 blocks](https://github.com/huggingface/diffusers/blob/0905b667fe3f37f43b6ef9c6e05ad20636206938/src/diffusers/modular_pipelines/flux3/modular_blocks_flux3.py)
declare `text2image` and `image_conditioned`. The selected MoDiff provider is
separately pinned; reading newer upstream code does not activate it. Current
official documentation describes references, layouts, prompt upsampling and
LoRA, not a compatible native Fill/ControlNet/IP-Adapter route for these selected
Klein checkpoints. See the [official introduction](https://huggingface.co/blog/flux-3).

| Capability | Current position | Delivery decision |
| --- | --- | --- |
| Text and ordered-reference generation | Existing generic image stages; accepted scoped examples | Preserve current recipes and switching behavior. |
| Dedicated PUP | Existing explicit Enable/Preview/Apply workflow; trained T2I evidence | Add the remaining reference-edit and layout-preservation acceptance scenarios without silently rewriting prompts. |
| Typed layouts and larger reference sets | Contracts exist; trained quality/resource evidence is incomplete | Validate actual authored geometry and reference identities, rather than counting coordinate text as typed-layout proof. |
| Native mask/Fill sampling | No declared native mask-consuming route in the selected upstream implementation | Requires an official compatible route or separately reviewed native extension. Do not present post-compositing as masked denoising. |
| ControlNet and IP-Adapter | No compatible native branch/checkpoint was established in this review | Requires model-specific trained weights, loading and denoiser integration; older FLUX.1 adapters cannot be assumed compatible. |
| Protected-pixel editing | Existing Crop/Resize/MaskComposite compositions | Offer clearly labeled edit-and-composite templates using ordinary nodes, with exact unchanged-region checks. |

This is partly an upstream/model availability boundary, not an omitted generic
ControlNet node. A working native adapter needs both a compatible trained artifact
and the inference path that consumes it. When those exist, add a small backend
adapter, declared execution specification, ordinary adapter controls and resource
accounting. Reuse existing Model Setup/Encode/Generate/Output stages. A
model-specific node is permitted when a real native contract needs it; adding
one cannot supply missing trained weights or an inference algorithm. Do not
accept arbitrary residuals or replace the official sampler without review.

Investigate mask-preserving sampling separately from trained Fill: an in-loop
latent-blending composition may be possible with reviewed official operators
without training another model. It still needs exact noise/state, packing,
mask/geometry and loop-ownership contracts plus real output checks. It is not
established by today's final-image composite or promised as native Fill support.

## MiniMax H3 Image: supported core and concrete gaps

MoDiff exposes text generation, one-reference edits and ordered multi-reference
edits through generic image stages. Its reviewed adapter preserves reference
preparation, generator continuation, requested geometry and minimum generation
area. The profile is currently Expert-only, native BF16, with declared NONE/SSD
group placement; these are not general Auto or hardware qualifications.

| Capability | Current position | Work to complete |
| --- | --- | --- |
| Text-to-image, image edit, ordered multi-reference | Existing; scoped full-model output evidence | Preserve order and upstream preparation. Remove the invented eight-reference adapter cap; forwarding nine inputs is a contract test, not trained nine-reference quality or fit qualification. |
| Six-step sampling, guidance and negative prompts | Correctly model-owned; arbitrary steps, CFG, negative prompts and strength are rejected | Preserve the model's fixed sampler and hide ineffective controls during switches. Their absence is not an integration defect. |
| Reference geometry and minimum generation area | Existing upstream preparation and explicit area control | Verify requested and effective rounded canvas dimensions independently. |
| Native prompt expansion, `MiniMax-H3-Image-PE` | Not integrated | Highest-priority model feature: add a reviewed local provider to the existing ancillary Preview/Apply flow. |
| Image-edit prompt expansion | Not integrated; the publisher distinguishes released T2I PE weights from an editing prompt recipe | Admit an explicit VLM-based editing recipe, preserve ordered inputs and test semantic preservation. Do not claim separately trained editing PE weights were released. |
| Official high-resolution landscape presets | The diagnosed output-area/side caps are removed in the adapter and fresh operation schemas | Test published sizes, native rounding and actual high-resolution output separately; retain the upstream bound on the minimum-generation-area setting. |
| Native ControlNet/IP-Adapter/Fill mask sampling | No compatible official route or trained artifact established | Track as an upstream dependency; use honest reference-edit/mask-composite alternatives where useful. |
| LoRA, other packed formats, native batch/distributed execution | Not declared by the reviewed H3 Image profile | Review each real upstream loader/format/placement independently; do not infer support from a generic component mixin. Use graph iteration for labeled serial variations. |

The [official Diffusers H3 Image guide](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3_image)
documents the core tasks, fixed sampler and sizing. The separate
[official PE release](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE/blob/35603d4b63949afa12cb098227fffefedf081afa/README.md)
provides text expansion with an aspect-ratio result. Its model card supplies an
editing recipe, describes Transformers/Diffusers and SGLang usage, and distinguishes
the T2I release from editing-specific training. Hosted Context-IR is a service,
not an automatically included local image-model feature. Video/audio MiniMax H3
is a different checkpoint/task family.

### First implementation package: local H3 prompt expansion

1. Review the immutable PE model/config, processor, publisher prompts, request
   defaults, two-field result schema and licence. The inspected configuration
   names `Qwen3_5ForConditionalGeneration` and ModelOpt-packed weights. Resolve
   actual official Transformers quantizer/kernel support on each intended host;
   do not treat FP4, NVFP4, NF4 and dense BF16 as interchangeable.
   Publisher memory examples are not proof of fit on either of our machines;
   in particular, the documented approximately 18–19 GiB GPU path does not
   establish fit on a 16 GB Windows GPU. Review a different explicit placement
   or the Linux path rather than silently changing the format.
2. Prefer the existing generic Hugging Face image/text loader/generator and
   ancillary lifecycle with a small model-specific backend adapter. Keep prompt
   assembly and output normalization backend-owned. Reuse declared reference
   inputs for editing. No SGLang/vLLM server or hosted API is introduced.
3. The documentation links `diffusers/minimax-h3-image-pe-blocks` and demonstrates
   remote-code loading. That Hub metadata endpoint returned 404 to this review
   request. Resolve availability and inspect an immutable source/dependency
   package before considering that route. Documentation is not executable
   availability or consent. Never enable `trust_remote_code` implicitly.
   The [publisher's merged documentation discussion](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE/discussions/2)
   explicitly states that the blocks repository is private. The reviewed
   Transformers 5.18 quantizer registry rejects the released `modelopt` config;
   its separate `nvfp4` quantizer is not a substitute for this packed checkpoint.
   Until a public, inspectable official loading/conversion path exists for an
   admitted host, local native PE is an upstream execution dependency. Keep this
   task open rather than adding an unusable provider or downloading unsupported
   weights. No silent dequantization, format substitution or hosted fallback.
4. Add the exact reviewed optional dependencies, if required, to the executable
   installation contract. Discovery/Auto inspection remain free of installs and
   model execution. Reuse valid existing licence authorization where applicable;
   do not invent a new acknowledgement or ask for an already satisfied decision.
5. Present **Enable enhancement → Preview → Apply** using existing generic
   controls. Keep original and proposed prompts distinct. Treat aspect ratio as
   a proposal: retain an explicitly chosen ratio and require explicit application
   of an adaptive suggestion. Model output cannot alter loaders, files or policy.
   Start with the selected plain-prompt contract. Refuse an authored typed layout
   unless that exact provider declares and proves its preservation; H3 does not
   inherit FLUX 3 layout support merely by sharing the enhancement controls.
6. Bind preview, model/processor identity, ordered image bytes, source workflow
   and current prompt/layout to Apply. Preserve Cancel, Undo/Redo, stale-result
   rejection and Save/reopen. Use current ancillary owner release and fresh
   memory checks before loading either the PE model or the image model.
7. Validate contracts and source packaging without weights first; then run
   text-only PE, reference-edit PE, invalid-schema, cancellation, warm reuse and
   subsequent image consumption on admitted hardware. Publish useful on/off
   comparisons only after acceptance. No hardware or Auto promotion follows
   from adding the menu offer.

### Second package: native size presets

The original H3 adapter capped requested area at 16,777,216 pixels. Published
4736×3552, 5024×3360, 5472×3072 and 6272×2688 presets slightly exceed that
bound; this was a diagnosed **MoDiff limitation**, not an upstream rejection.
Requested width/height now accept positive graph-safe integers, with no invented
8,192-pixel side or fixed output-area ceiling. Upstream owns canvas rounding and
the returned image is resized to the requested dimensions. The distinct
`min_generation_area` parameter retains its actual upstream 1,024–16,777,216
range. A fresh RAM check rejects an output or rounded canvas whose RGB buffer
alone exceeds available memory; that necessary floor is not an activation-memory
estimate, Auto fit or a hardware qualification. Existing resource/file budgets
continue to apply. Preserve existing saved settings and accepted images. A
limit change needs no blanket rerender of the original 89, but new high-resolution
quality/resource claims require their own live runs.

## Canonical suite matrix

Each row is a task family, not a promised single model or a new completion count.
Split its materially different native tasks into explicit recipes. Reuse existing
accepted cases where their actual settings and evidence match.

| ID | Workflow family and required useful templates | Current position / main work |
| --- | --- | --- |
| S01 | Text and reference-aware prompt expansion; explicit native PUP/PE on/off comparison; typed-layout preservation | Existing FLUX path; extend H3 and finish semantic/layout evidence. |
| S02 | Text-to-image, aspect ratios, seeds, batches and fair cross-model comparison | Existing; match actual inputs and document model-specific samplers. |
| S03 | Single/ordered multi-reference editing, identity consistency, multi-angle subject and scene composition | Existing scoped models; qualify each reference count and preserve input roles/order. |
| S04 | Native inpaint/outpaint, object removal/replacement and protected-region editing | Existing on supported models; distinguish native conditioning from final compositing. |
| S05 | Product/material transfer, relighting, restoration of a scene and reference-guided redesign | Existing recipes plus quality/evidence work; exact before media and unchanged-region review. |
| S06 | Posters, typography, multilingual text, spatial layout and brand/product composition | Existing examples; add typed-layout and exact-text checks. |
| S07 | Edge/depth/pose/segmentation/normal controls and multiple structural conditions | Existing scoped ControlNet/control routes; review each model/control combination and new preprocessors. |
| S08 | LoRA stacks, image-prompt adapters, Redux and combined structural/style/reference control | Existing model-specific paths; exact adapter ownership, scales, compatible weights and terms remain mandatory. |
| S09 | RGBA layer decomposition, background/subject isolation and editable recomposition | Existing Qwen route; alpha, fidelity, occlusion, shadows and independent layer movement checks. |
| S10 | Deterministic and learned image upscale, tiled detail, denoise/deblur/restoration | Existing resize/upscaler paths; separate resampling from learned restoration and generative detail. |
| S11 | Automatic image segmentation, temporal object tracking, background removal and matting | Review automatic mask/model routes; current mask compositing alone does not establish these. |
| S12 | Numeric depth, normals, pose/line-art/edge preprocessing and visual previews | Depth routes exist; normals/other estimators need admission and useful quantitative outputs. |
| S13 | Text-to-video and still-image animation with purposeful motion | Existing scoped video paths; complete full-recipe cases and judge motion, not just valid encoding. |
| S14 | First/last keyframes, clip continuation and controlled transitions | Existing/experimental contracts; qualify anchor preservation, frame/timebase identity and complete clips. |
| S15 | Video-to-video, remaster, masked replacement and video outpaint/reframe | Existing selected paths; close retained recipe/Auto/mask/upscaler issues and verify timing. |
| S16 | Character animation/replacement, pose/motion transfer and camera-conditioned video | Existing selected/experimental routes; exact model support and real motion/identity examples required. |
| S17 | Audio-driven lip-sync and natively synchronized audiovisual generation | Mixed contract/review state; distinguish joint generation, speech-driven motion and ordinary muxing. |
| S18 | Multi-shot stories, long sequences, lyric/music videos and resumable production | Existing composition/sequence tools; resolve complete owner/storage contracts and accepted-recipe mismatches. |
| S19 | Learned frame interpolation, temporal restoration and video super-resolution | Reviewed tiled upscale exists; deterministic frame blending is not learned interpolation. New models/runtime need review. |
| S20 | Text/lyrics-to-music, instrumental/vocal choices and reviewed music LoRA | Existing ACE and other selected routes; complete current music-model/rights cases. |
| S21 | Source-audio variation/cover, audio repaint and extension | Existing partial routes; prove the actual codec/source-conditioning task rather than substituting ordinary generation. |
| S22 | Single/multi-speaker TTS, voice conversion and timed voiceover/dubbing | Review eligible local libraries/checkpoints, voice consent and audio/timestamp contracts. |
| S23 | Transcription, translation, aligned subtitles and speaker diarization | ASR/translation contracts exist; add accepted examples and separately review diarization. |
| S24 | Stem separation, vocal/instrument isolation, noise cleanup and sound effects | Review trained routes; preserve sample rate, channels, duration and leakage/quality evidence. |
| S25 | Text/image-to-3D, multiview reconstruction, mesh/point-cloud output, textures/UV and orbit preview | Establish existing ShapE examples first. Modern Hunyuan3D/TRELLIS-like routes require eligible library admission, not just a Hub download. |
| S26 | Captioning/VLM, visual questions, OCR/layout extraction and structured local text workflows | Existing generic understanding/text routes; review task-native support and accepted numeric/text/media outputs. |
| S27 | Crop/resize/fit/stitch, masks/composite, colour, trim/FPS/timebase, subtitles, mux, import/export and metadata | Existing deterministic tools; use bounded media fixtures, exact semantics and native file-boundary tests. |
| S28 | Lists/batches, reusable Blocks, saved workflows, comparison grids, multi-model chains, queue/cancel/recovery | Existing graph machinery; accept real compositions with owner release, repeatability and persistence, not another executor. |
| S29 | Native vector generation and vector-oriented design | Current reviewed Comfy vector examples are API routes; excluded from local parity until an eligible local path is explicitly reviewed. Deterministic SVG work is separate. |
| S30 | Hosted partner APIs and external inference services | Excluded by the current local-only runtime boundary. A future change requires a separate product/security decision. |

For each supported structural-control family, include separate edge/depth/pose
cases instead of treating one successful adapter as proof of all controls.
Likewise, separate first/last keyframes, motion transfer, character replacement,
lip-sync and joint AV. A mask input, speech input or compatible Python signature
does not establish those tasks.

### Application templates

Build these as compositions of the canonical families, without new model nodes:

- Product packshot → material/label variation → contextual advertisement.
- Brand/poster layout → exact multilingual typography → output variants.
- Consistent character → multi-angle/storyboard → controlled animation.
- Garment/reference fashion edit with identity, garment and protected-region
  checks; distinguish native virtual try-on from a general reference edit.
- Architecture/interior redesign with structural control and preserved geometry.
- Song/voice → timed scenes/lip-sync → captions/subtitles → final AV export.

## Delivery order and dependencies

The current execution priority is FLUX 3/H3 feature completion and its tests
before resuming the outstanding original Gallery cases. Packages 1A/1B and the
attainable native FLUX 3 control review therefore precede the remaining Phase 0
execution. This reorders work; it does not change the original 89-case denominator.

| Phase | Work package | Exit criterion |
| --- | --- | --- |
| 0 | Finish the original 89: register accepted exact-current cases, honestly version accepted revisions, close retained-evidence joins and complete admitted full jobs | All 89 distinct original rows satisfy their recorded current recipe, quality, rights, provenance and public-media checks; unresolved rows retain a reason. No denominator substitution. |
| 1A | H3 PE provider and explicit Preview/Apply integration (S01) | Immutable local model/runtime admitted; trained T2I/edit-recipe and image-consumption evidence; safe Apply/persistence/resource lifecycle. |
| 1B | H3 native-size presets and remaining FLUX layout/reference/PUP evidence (S01–S03/S06) | Documented source limits corrected where justified; actual new workload evidence and honest platform boundaries. |
| 2 | Existing-but-unaccepted depth, understanding, ASR, 3D and utility scenarios (S12/S23/S25–S28) | Useful outputs and native workflow/media examples, with no library addition where existing eligible routes suffice. |
| 3 | Automatic masks/normals, advanced character/keyframe/AV video and audio processing (S11/S14/S16/S17/S22/S24) | Runtime/checkpoint eligibility first; reviewed generic ports/adapters, complete native execution and representative quality acceptance. |
| 4 | Learned interpolation/restoration and modern reconstruction (S19/S25) | Eligible official-library implementation plus resource/security contracts; no substitution of blending, upscaling or text-to-3D for the intended task. |
| 5 | Application compositions and broader hardware/resource variants | Reusable polished templates with source-bound examples; qualification attached only to actually tested placements/hardware. |

Resume outstanding Phase 0 generation after the feature checkpoint. Keep one
active and one incoming model family per staging
volume. Windows handles fitting smaller work, tests, evidence and publishing;
Linux handles admitted large jobs. Actual capacity determines placement.
Rights, unresolved capabilities and unapproved paid cloud use remain real holds.
Do not repeat unchanged failures or delete a dependency needed by an open case.

## Definition of done and proof preservation

For every canonical task and materially different recipe:

1. **Declare:** task inputs/outputs, exact official library/model/adapter pins,
   provenance, terms, file selection, resource recipe and intended limitations.
   A Hub repository alone does not establish eligible execution code.
2. **Implement:** reuse generic stages, file/security boundaries, Auto/Expert,
   cancellation and the single graph executor. Model details belong in backend
   adapters. No library-owned graph, browser runtime or silent package install.
3. **Test without weights:** relevant normalizers, official API/processor contracts,
   field bounds, invalid inputs, graph ownership and metadata. Add regression
   tests for diagnosed defects; then run the required compatible repository gates.
4. **Exercise the app:** create, switch compatible models/tasks, review incompatible
   ports, edit fields, expand/collapse stages, Cancel/Undo/Redo, save/reopen and
   run. Inspect legible wide/compact layouts and the actual output preview.
5. **Prove native output:** retain actual consumed controls, effective geometry,
   input pixels/hashes, model/runtime/source identities, resource observation,
   output hashes, complete media and quality findings. Mark absent historical
   fields absent. Do not relabel authored requests as measured consumption.
6. **Accept and register:** bind user acceptance/rights to the exact media and
   recipe; use the ordinary Gallery verifier and immutable public Dataset.
   Preserve earlier accepted assets and explicit licence holds. A comparison
   showcase must identify its own recipe rather than qualify different defaults.
7. **Qualify separately:** attach Auto/hardware/precision/placement claims only
   to their own passing evidence. Contract, HTTP, tiny-model, full-model,
   user-accepted and registered states remain distinct.

Maintain an impact matrix. A presentation-only grouping change must preserve
expanded executable graphs and needs UI/persistence checks, not 89 new renders.
A backend sampler/conditioning change needs focused regression and affected
native-route evidence. A new model/adapter or resource recipe needs its own
admission. Unchanged accepted media and source-bound historical records remain
valid within their original scope.

See [image stages](image-pipeline-stages.md), [video stages](video-pipeline-stages.md),
[audio workflows](audio-workflows.md), [Auto memory](workflow-auto-memory.md),
[FLUX 3 adapter boundaries](flux3-adapters.md),
[runtime support](runtime-support-matrix.md) and
[required engineering procedure](cluster-engineering-lessons.md).
