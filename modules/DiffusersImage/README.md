# Ordinary Diffusers image nodes

`LoadPipeline` and the generic image actions execute reviewed ordinary Diffusers
pipelines. An ordinary catalog Block contains these real nodes and media loaders /
preview. It is not an upstream Modular Diffusers hierarchy. Native Modular routes
remain in `modules.ModularDiffusers`.

## FLUX adapter boundaries

- `FluxControlNetPipeline`, `FluxControlNetImg2ImgPipeline` and
  `FluxControlNetInpaintPipeline` load a separate, pinned `FluxControlNetModel`.
  Source, control and mask inputs remain distinct. Scale and control start/end
  are exposed. Single, multiple and Union conditioning use the explicit Control
  Component contract below. Config incompatibility fails before base weights load.
- `Flux2KleinKVPipeline` uses its own exact KV checkpoint and attention-processor
  ownership. It is not interchangeable with ordinary Klein through `from_pipe`.
  Text-only, edit and multi-reference contracts expose only actual upstream controls;
  KV has no guidance/strength call argument. Reference inputs require at least 64px.
- Five legacy FLUX actions retain their existing Guidance → `true_cfg_scale`
  mapping. A separate explicit override controls distilled guidance. Disabling or
  omitting the override preserves old calls. ControlNet T2I instead exposes distilled
  guidance primarily and true CFG secondarily. Do not infer equivalence from labels.

All model/adapter files use the existing Hugging Face Hub resolver and immutable
catalog revisions. Optional-runtime, resource, trust and local-file boundaries
are unchanged. No downloads occur merely from catalog discovery.

## Optional typed call inputs

Ordinary FLUX actions expose the selected pipeline's reviewed optional inputs as
normal sockets. These include alternate prompts, images per prompt, sigmas,
Generator(s), starting latents, prompt/pooled/negative embeddings, masked latents,
reference images, text-encoder output layers and Kontext resize / FLUX.2 caption
temperature controls where upstream declares them. Connect ordinary value, tensor,
generator or media nodes; Configure Interface can expose a socket on its Block.
Disconnected sockets do not change existing parameters or upstream defaults.

Connected prompt embeddings take precedence for that call only: the authored
prompt stays saved, while the upstream text argument becomes `None`. FLUX.1 pooled
and sequence embeddings must be connected together. Supplied Generator objects are
used without reseeding; serialized objects are rejected. Sigmas determine actual
progress length without rewriting the saved Steps field. Batch safety bounds are
eight images and sixteen megapixels total. Switching pipeline hides unsupported
sockets but does not erase saved values; incompatible connected inputs fail with
their field name rather than being silently ignored.

Use the normal **Seeded Generator** primitive for a reproducible connected
Generator: it creates a fresh object on each graph execution, rather than reusing
a previously advanced cached RNG. Intentional sharing among consumers within
the same execution is retained. CPU generators work with Diffusers accelerator
noise preparation; choose an explicit available device when a consumer requires it.

Group offload covers direct VAE encode/decode entry points with leaf hooks.
For VAEs with directly read BatchNorm statistics, the small VAE uses synchronous
CPU offload even under the disk strategy; the large transformer/text encoders
retain disk offload. This prevents uninitialized normalization buffers and black
images without changing precision, prompts or inference settings.

IP-Adapter loading/call inputs and multi-ControlNet/Union are implemented and
covered by actual tiny upstream runtime tests; full-weight acceptance remains
route-specific in the runtime support matrix. These controls do not invent a Modular
hierarchy for ordinary pipelines. App-owned progress/cancellation callbacks and return structure
remain managed by MoDiff, not arbitrary serialized Python callables.

## Explicit latent outputs and decode

The sixteen reviewed ordinary FLUX pipeline classes offer `output_type=latent`
and a separate typed **Latents** output. Images and pixel dimensions are empty
in this mode; the exact upstream tensor is retained. Existing PIL defaults are
unchanged. Non-PIL inpaint results use the upstream result without MoDiff's
PIL-only mask compositing; `padding_mask_crop` still requires PIL output.

Add **Decode Image Latents** from the normal Diffusers Image category. Connect
the same loader's Pipeline and the generating node's Latents, set the original
Width/Height, and connect its Images to Preview. It validates the layout before
using the matching upstream unpack/VAE/image-processor path. FLUX.1 returns
packed normalized tokens; FLUX.2 returns denormalized, unpatchified VAE latents.
A returned FLUX.2 tensor is **not** directly interchangeable with its starting
noise input. Arbitrary tensor edges do not silently transform layout or scaling.

## Definitions and evidence

Update execution specifications, selected `image_contract` fields, dependency pins,
composite admissions and paired compiled V2 identities together. Existing saved
definitions and instance values are not silently rewritten. New backend-defined
routes use their compiled defaults when no explicit legacy Studio profile exists.

Tests distinguish contract fixtures, tiny real components and full-weight frontend
generation. `test_flux_controlnet_runtime.py` uses actual small ControlNet/transformer/
VAE components but fixed embeddings; `test_flux_klein_kv_adapter.py` verifies real KV
cache lifetimes. Neither proves full model quality. For family limitations and
the separate acceptance gates, see the
[runtime support matrix](../../docs/runtime-support-matrix.md#model-families-and-support-boundaries).

## Optional ordinary FLUX inputs

Disconnected optional sockets do not override saved defaults. Redux accepts
independent `prompt_embeds_scale` and `pooled_prompt_embeds_scale` scalars or
per-reference lists (1–8 items, finite values from -100 to 100). A connected list
must match the reference count. Each overrides only its own prior argument;
omitting either retains its existing secondary-reference-strength behavior.
The prior also accepts `prompt_2` and paired `prompt_embeds` /
`pooled_prompt_embeds`. These feed the Redux prior, not the base generator: its
fused embeddings are what the base consumes. Supplying embeddings leaves the
saved prompt unchanged and omits text only for that call. Historical contracts
without these optional ports remain valid; no existing instance is reseeded.

FLUX.1 uses `joint_attention_kwargs`; FLUX.2 uses `attention_kwargs`. Reviewed keys
are `scale` for LoRA scaling and `attention_mask` for a materialized boolean or
additive floating-point Tensor. The generic **Attention Arguments** node builds
this object and can be added to any compatible workflow or Block. Unknown keys,
callables and KV-cache injection are rejected explicitly. Klein KV rejects masks
because its reference-cache path does not apply them. The mapping is copied, but
the tensor is not reshaped, moved or mutated. Attention scaling does not load an
adapter. See [Attention Arguments](#attention-arguments) for wiring,
shape/device requirements, limitations and proof levels.

## Image-prompt adapters

`Image Prompt Adapter` configures exact local Hub files; it does not modify a
resident model. Connect its output to `Load Diffusers Image Pipeline` → **Image
Prompt Adapter**. That loader attaches the genuine upstream adapter to its own
new transformer before offload hooks. Other loaders and unchanged saved workflows
are unaffected. Missing files fail before base-model allocation; corrupt files
fail their SHA-256 checks. Installation remains an explicit Model Manager/Hub pull.

The selected ordinary FLUX, img2img, inpaint, Kontext, Kontext-inpaint and
ControlNet-T2I actions expose upstream positive/negative reference-image and
embedding sockets. These are not supported by FLUX.2 or ControlNet edit/inpaint.
Connect references to those action sockets, not to the configuration node.
The independently pinned CLIP vision encoder is needed for image references;
embedding-only operation needs both positive and negative tensors because upstream
otherwise encodes a blank image for the missing side. Ordinary pipelines do not
gain fictitious Modular internals from this feature.

Chain up to four configuration nodes using **Previous adapters**. They share the
same selected vision encoder and preserve adapter order. Scalar strength and an
optional list of per-transformer-layer scales follow upstream semantics. Editing
configuration rebuilds that loader's owned pipeline; there is no in-place mutation
of a cached sibling's attention processors. Runtime shape errors remain explicit.
The current default artifact is the XLabs FLUX IP-Adapter with its separately
pinned CLIP encoder. Model licenses still apply. Tiny-component tensor, dispatch,
scale and rollback tests are separate from full-weight frontend acceptance.

## Control Component composition

`Control Component` configures an immutable Hub repository/revision without
loading weights. Chain up to four through Previous components, then connect
Control Components to the existing pipeline loader. This explicitly replaces
the loader's single conditioning selection; disconnecting restores that selection.
The loader owns fresh `FluxControlNetModel` objects and the genuine upstream
`FluxMultiControlNetModel`. No shared resident model is mutated.

The three ordinary FLUX ControlNet actions support per-condition scale/start/end
lists and a typed `control_mode` input. Modes are validated against the actual
Union model's `num_mode`. Shared Union Conditions uses one Union model for several
images/modes; distinct components require matching image counts. Mixed raw-pixel
and VAE-latent preparation is rejected because the pinned upstream preprocessor
uses one shared strategy. Existing scalar defaults are unchanged. Errors identify
the affected count, field or condition before expensive inference.

These ordinary pipelines do not expose an invented Modular Diffusers hierarchy.
The configuration and action are normal nodes and can be saved inside User Blocks.
Current tiny real-runtime tests cover all three actions, single/multiple/Union
composition, normal node dispatch, repeat execution and inpaint mask preservation.
Full-weight native acceptance uses the separate
[qualification gates](../../docs/runtime-support-matrix.md#acceptance-and-remaining-qualification).

## Using FLUX Blocks

These interactions apply to the ordinary entries described above and the
[native Modular entries](../ModularDiffusers/README.md).
See the [runtime support matrix](../../docs/runtime-support-matrix.md#model-families-and-support-boundaries)
for family limitations and qualification boundaries.
This guide describes supported interaction patterns, not guaranteed prompt fidelity
or automatic publication approval.

### Choose the appropriate entry

| Need | Entry / behavior |
| --- | --- |
| Fast generation with genuine editable internals | **Flux2 Klein — Text To Image**, distilled model |
| Non-distilled Klein | Klein Base; keep its own suggested steps and guidance |
| Instruction/reference editing | Kontext, FLUX.2 or Klein edit entries; supply the requested reference images |
| Precise edit region | An inpaint entry plus an explicit mask; white is editable, black is preserved |
| Extend a canvas | Outpaint; fresh forms use strength 1 to fill newly blank borders |
| Edge/depth structure | Canny/Depth control entries; provide the corresponding control representation |
| A separate ControlNet adapter | ControlNet entries plus compatible control components/conditions |
| Visual reference variation/blending | Redux; reference influence is not the same as exact instruction editing |

**Fill outpaint requires both a prepared wider canvas and its mask.** The mask
marks new/replaceable regions white and the retained center black. Match the
instance dimensions to that canvas. The separate Kontext/Klein outpaint entries
prepare their canvas automatically; do not assume every outpaint route has the
same input contract.

A required image or mask with neither a selected file nor an enabled incoming
connection blocks Run, including in Expert mode. **Fix → Open required input**
selects the affected Block; choose the file or connect a compatible producer.
This action does not invent a mask, reset other fields, or rewrite the graph.

The **Standard Diffusers** entries wrap ordinary upstream pipelines with real
component/data inputs. They do not pretend that an ordinary pipeline provides
the same nested hierarchy as a genuine **Modular Diffusers** entry.
Likewise, Canny/Depth's dedicated model weights are not interchangeable with a
separate ControlNet adapter just because both constrain image structure.

### Edit and reuse

1. Add the entry from Nodes. Change a prompt or parameter while collapsed;
   expanding must preserve those values and connections.
2. Expand one level at a time. Select a child to resize it; **Advanced** exposes
   secondary controls. **Arrange** changes layout, not the execution recipe.
3. For a compatible model switch, use the loader's **Model** selector. It only
   offers admitted alternatives for that task. Independent saved route drafts
   are retained; switching does not turn a text-to-image composition into edit.
4. Add a normal node from the palette and connect compatible ports. Tensor,
   component, state and loop requirements still apply. A broken draft should
   remain editable; inspect the affected node/port and **Fix** suggestions.
   Review a repair before applying it; Undo reverses the edit. Not every invalid
   tensor shape or arbitrary composition has an automatic repair.
5. **Save** the workflow to retain its instances. To reuse a root or nested Block,
   select its save action and **Save as new User Node**. Reinsert it from User
   Nodes; edits to the copy must not change the original or library defaults.
6. Run, then inspect the Gallery's backend-captured settings and references.
   A completed execution and a faithful visual edit are separate checks.

[Attention Arguments](#attention-arguments) is a normal reusable node for
supported attention masks and optional LoRA scale. It does not load a LoRA.
Connected tensors retain their real dtype/device/shape; the graph does not
silently reshape incompatible inputs or execute arbitrary Python callbacks.

### Preserve deliberate settings

Existing saved outpaint strength values are not silently replaced with 1.
Change an old instance explicitly if blank border regeneration is intended.
Do not copy a base model's step count onto a distilled model, or vice versa.
Model/resource changes, reduced steps and quantization must remain explicit.
Model installation uses the app's Hugging Face Hub path and pinned revisions.

## Attention Arguments

`modules.Tensor.AttentionArguments` is an ordinary, reusable node in the Primitive
category. It builds a data-only options object for a compatible attention input;
it does not load models, adapters or executable processors.

1. Add **Attention Arguments**, either to a workflow or inside an expanded Block.
2. Optionally connect a boolean or additive floating-point **Attention Mask**
   tensor. Materialized, nonempty rank 2–4 tensors are accepted; additive negative
   infinity is valid. The mask must broadcast to the consumer's attention scores
   and have a compatible device/dtype. No implicit resize, cast or device move is
   performed. NaN, positive infinity, sparse/meta tensors and JSON arrays are not
   valid masks.
3. Enable **Set LoRA Scale** only when a call-level scale override is wanted.
   With it disabled, the scale is omitted, even if the numeric field was edited.
   A scale does not install an adapter or affect a model with no active LoRA.
4. Wire **Attention Arguments** to the supported `joint_attention_kwargs`
   (FLUX.1/Modular FLUX) or `attention_kwargs` (FLUX.2) input. Save the workflow or
   enclosing User Node normally; tensor values are recomputed by their producers,
   not serialized as model data in the workflow.

Ordinary FLUX adapters accept the reviewed `scale` and `attention_mask` keys.
An empty object preserves upstream defaults. Unknown keys, callbacks, processors
and mutable KV execution state are rejected with the affected input name.
Klein KV does not apply an ordinary attention mask during its reference-cache
path, so that combination is explicitly rejected rather than silently ignored.
The pinned FLUX IP-Adapter processor declares `ip_adapter_masks` but does not use
it; this is not exposed as an effective control. Its ordinary `attention_mask`
path is covered separately by actual upstream execution tests.

The builder and call boundary retain the supplied tensor's identity. Attention
shape/device failures remain upstream errors: fix the producer's dimensions or
device, or disconnect the optional mask. They do not justify replacing prompts,
models, seeds or unrelated connections.

### Attention verification

Contract tests cover ordinary-node dispatch, invalid values, preserved tensor
identity, omitted scales and unsupported KV inputs. Tiny-real pipeline tests
compare exact adapter/direct-upstream outputs and verify that a nontrivial mask
changes the output across ordinary, ControlNet/IP-Adapter and Modular FLUX paths.
These are semantic tests, not full-weight image-quality approval. Native palette/adoption/wiring/persistence, loaded-LoRA effects, and full-weight
generation require their own [acceptance evidence](../../docs/runtime-support-matrix.md#acceptance-and-remaining-qualification).
