# Image pipeline stages

The generic `DiffusersImage.LoadPipeline`, `EncodeInputs`, `Denoise`, and
`DecodeLatents` nodes run selected image pipelines through the ordinary MoDiff
graph. The visible sequence is **Model Setup → Encode Inputs → Denoise → Image
Output**. Source images, masks, references, and Preview remain real connected
nodes. The stages call the official Diffusers APIs; they do not replace its
denoising loop or create a second pipeline owner.

Stage eligibility comes from a declared backend adapter and the current selected
execution profile, task, repository, and immutable revision. The loader records
the actually loaded artifact pair and components. Local, compatible-repair, or
advisory selections can continue using their existing whole-pipeline actions
without receiving stage eligibility. Old whole-pipeline graphs and raw latent
decoding remain available. New staged latents require their matching binding.

## Inputs and state

`EncodeInputs` returns `image_conditioning` owned by the selected loader. Its
material is sealed against changes to the owner, component identities, runtime
recipe, adapters, geometry, and relevant input values. `Denoise` consumes that
conditioning and returns real latents, a matching `image_latent_binding`, and the
actual width and height. `DecodeLatents` consumes all four outputs and the same
pipeline owner. Changing a connected owner or material invalidates the cached
stage output before reuse.

The encoder performs prompt-batch expansion once. Denoise consumes the resulting
expanded batch with an images-per-prompt count of one. The requested count belongs
to Encode Inputs. Geometry and sequence-length constraints remain bound where
the official encoding depends on them. The existing image decode limits apply
before denoising, including a maximum batch of eight and a maximum decoded batch
area of 16,777,216 pixels.

The executed stage records the values it actually consumes. Oversized consumed
text is explicitly omitted under the existing bounded provenance rules; it is
not replaced with invented `null` values. Inputs supplied as embeddings can have
an actual unused prompt value of `null`.

## Selected strategies

Fill uses its official text encoding in Encode Inputs. Its source image and mask
remain sealed material. The official latent pipeline
call still performs VAE/mask preparation during Denoise. Image Output decodes
with the actual VAE and retains MoDiff's existing PIL source/mask composite.
Changing the denoise seed can reuse the text encoding. The existing
`reference_strength` authoring control and defaults remain available for graph
compatibility; Fill has no corresponding official parameter and this strategy
does not consume it. Redux consumes its own reference-strength input.

Control encodes its text and control image using the official public preparation,
VAE, and packing methods. That encoding consumes the seed, so a changed seed
requires new encoding. The sealed post-VAE generator state is restored for each
denoise call. This preserves repeated execution and the official full-call batch
semantics.

Redux runs the official prior once in Encode Inputs, preserving reference order
and individual prompt/pool weights. The immutable prior and base repositories
remain one loader-owned bundle. The base gives its text encoders and tokenizers
to the prior and clears its own text ownership. Denoise calls only the base with
the prior's sealed embeddings and requests latent output. Image Output uses the
real base VAE and latent geometry. A changed denoise seed can reuse prior output.

External Generator objects and supplied latent or masked-latent inputs are
refused for these stage strategies until their state and continuation contracts
are separately reviewed. They are not silently discarded. Existing whole
actions retain their separate input contracts.

## Adapters, cancellation, and cleanup

The ordered adapter chain supplies all three stages from the same final pipeline
output. Authoring still requires current published authority and exact selected
policy; a saved hint does not grant permission. Shared, foreign, missing, or
partially wired consumers remain refused.

Redux's dense LoRA bridge uses the official loader methods on the base
transformer and the actual prior CLIP text encoder, with the corresponding real
pipeline owner for each. It never forwards text keys to the base's cleared text
encoder. Its initial contract accepts canonical complete dense Linear A/B pairs,
including both actual component inventories, and ordered bounded scalar scales.
Unknown prefixes, T5 targets, DoRA/normalization variants, incomplete pairs,
shape expansion, and metadata variants are refused before mutation. Real split
adapter mutations and reset currently require **NONE placement**, no active
quantizer, and a recorded CPU or single resident CUDA loading map. The bridge
also refuses actual installed Accelerate or Diffusers hooks across either
owner's complete component trees before unload or injection. Hooked target
recreation is a separate qualification boundary. An empty complete Redux owner
can be reused at its existing placement without changing its epoch or hooks.

Every successful adapter mutation creates a new owner epoch. A partial official
failure poisons the old encoding until complete reviewed reset or a new
successful loader. Reusing a loader after actual adapter unload also invalidates
the affected material. Empty no-adapter reuse preserves valid material.

The stages use the ordinary progress callbacks and Stop handler, and the
official pipeline hook finalizers. Cache release keeps actual models weakly
referenced by stage records, so independent image output can survive owner
teardown. Existing Auto/Expert admission, resource budgets, file containment,
artifact hashes, and cancellation boundaries remain in force.

Tiny CPU component and dispatcher tests establish numerical and lifecycle
contracts. They do not establish trained-artifact GPU fit, Windows memory
headroom, disk-hook adapter mutation, or live recovery qualification. Those
checks use the ordinary application and original settings separately.
