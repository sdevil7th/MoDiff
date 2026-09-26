# Exact image variant review

Reviewed against Diffusers `fbf49e7f35857f76bc57b177e26f12b03687c668`.
These are implementation decisions, not live-image or resource qualification.
The source-bound machine-readable decisions are in
`modiff/image_native_variant_reviews.py` and the generated readiness ledger.
Existing standard profiles and saved graphs are retained.
Exact operation recipes carry `operation_recipe: true`: they remain selectable
in the operation picker but do not replace legacy Auto model/task owners.
The new ordered-reference recipes are ordinary operation graphs, not new
compiled Cluster admissions.

## Native PAG

Select `sdxl-pag:modular` on the ordinary SDXL Modular task picker. Text-to-image,
image-to-image, inpaint, ControlNet text-to-image and ControlNet image-to-image
use the existing editable stages and one shared official PAG Guider. The explicit
Layers node targets all ten mid-block transformer layers and can be edited.
CFG scale defaults to 5, PAG scale to 3, and the native
perturbation interval to 0–1.

This is an alternative, not a silent conversion of the standard PAG executor:
the pinned native guider has an **exclusive start boundary**, so step zero is
unperturbed, and has no adaptive-PAG-scale argument. One-step runs therefore do
not perturb attention. Use the retained standard profile when those semantics
are required. The native recipe defaults to 50 steps. Tests exercise the actual
upstream attention hook, perturbation, numerical combination and hook removal
using tiny CPU tensors; they do not establish visual quality of pretrained weights.

## Twelve exact variant decisions

| Advertised route | Decision |
| --- | --- |
| FLUX dev image-to-image | Native `flux-dev:modular/image_to_image`; standard `edit_image` is the task alias, not Kontext editing. |
| FLUX dev FP8 image-to-image | Artifact blocked: root checkpoint has no reviewed component-layout/conversion recipe. |
| Kontext multi-reference | Native `flux-kontext:modular/multi_image_reference_edit`; visible Stitch Images operation constructs the same ordered equal-height RGB canvas as the standard adapter. |
| Kontext NVFP4 multi-reference | Artifact blocked: root checkpoint is not an interchangeable full component repository. |
| FLUX Krea text-to-image | Native `flux-krea:modular`, exact Krea artifact, 28 steps, guidance 3.5. |
| FLUX Schnell text-to-image | Native `flux-schnell:modular`, four steps, guidance 0, 256-token prompt default/limit in the starter. Upstream reads guidance-embedding support from the actual transformer config. |
| FLUX.2 dev multi-reference | Native ordered-reference task; separate latent/token ranges and reference position IDs, not image batching or a canvas. |
| Klein 9B KV text-to-image | Native `flux2-klein-kv:t2i-modular`. The exact cached index at `a6dfb36eca3a3906eb2fd460795adfb844e5fcce` declares distilled `Flux2KleinPipeline` and standard component classes. The upstream no-reference branch does not use KV caching. No 4B memory proof is inherited. |
| Klein 9B KV edit | Retained standard exception: modular Klein has no reference-K/V extract/cached loop at this pin. |
| Klein 9B KV multi-reference | Same exact upstream limitation; per-generation KV state is not cross-run caching. |
| Klein 4B multi-reference | Native ordered-reference task using the exact distilled 4B profile. |
| SDXL Turbo text-to-image | Native `sdxl-turbo:modular`, 512px, one step, CFG 0, repository scheduler and exact fp16 files. |

Reference recipes allow at most eight images. Horizontal composition checks input
and output pixel limits before allocating a canvas. FLUX.2 preserves individual
reference ordering. No license acceptance, artifact conversion, library upgrade,
new executor, or browser-side model-specific branch is implied.

## Verification boundary

Use the focused native PAG/reference, operation starter, optional runtime, image
processing and readiness tests when changing these adapters. Run client operation
contract/parser/export tests for their authoring projections. A backend-only fix
does not require the entire browser suite. Batch the final shared backend gate;
run only relevant browser journeys when actual UI behavior changes.

Missing downloads, blocked checkpoint layouts, hardware limits and image quality
remain separate from reviewed integration. Never turn a static review, a tiny
CPU attention test, or a base-model output into an exact pretrained-model pass.
