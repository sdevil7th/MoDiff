# SoHo fashion editorial demo

Execution success,
different image hashes and a large graph are **not** visual acceptance criteria.
The intent is an attractive, detailed editorial photograph and an unmistakable,
explainable change in every chapter.

## Presentation structure

| Chapter | Creative purpose | What the graph must demonstrate |
| --- | --- | --- |
| 01 | Adult woman in an ivory suit on a detailed SoHo street | Native text encoding, denoising and decoding |
| 02 | Opposed warm/cool lighting treatment | Editable custom image processing, before/after |
| 03 | Replace the suit with an emerald evening dress | Region masks, downstream generation, protected compositing |
| 04 | Replace the burgundy bag with metallic gold | Localized image-conditioned editing |
| 05 | Elaborate flower boutique | Larger architectural-region replacement |
| 06 | Vintage convertible instead of the taxi | Object replacement with the subject protected |
| 07 | Autumn foliage and fallen leaves | Coordinated environment changes |
| 08 | Rain and wet-street reflections | Material and weather transformation |
| 09 | Blue hour, illuminated windows | Time-of-day editing |
| 10 | Neon-lit night editorial | Strong atmospheric transformation |
| 11 | Monochrome editorial | Deterministic saturation control |
| 12 | Whole-pipeline alternative | Custom processing retained at the image boundary |
| 13 | Native return and final selected look | Preserved settings, connections and refresh persistence |

Before presenting a locally curated set, verify each chapter's own retained
image after reopening and refresh. Check executable values, edges and memory
policy as well as the photograph. A different file hash is not visual acceptance.
For transfer and hardware boundaries, see [Windows demo setup](windows-fashion-demo.md).

## Custom code with a visible purpose

Stage [`EditorialRegions`](../examples/custom_nodes/EditorialRegions/README.md)
through **Nodes → Custom nodes**, inspect its source and explicitly enable it.
It uses the normal node executor and existing Pillow/NumPy dependencies.

**Editorial Regions** produces an RGB image and a grayscale mask. Its polygon
list supports `add`, `subtract` and `intersect`, normalized coordinates and
feathered edges. Exposure, split color temperature and saturation are separate
controls. A subtract region can protect a face or hand. This is manual image-space
art direction, not automatic object segmentation or physical relighting.

**Protected Editorial Composite** blends the generated image over the original
using that same mask. Black-mask pixels remain exactly the original RGB pixels.
White-mask pixels use the generated edit. Inspect transitions at feathered edges.
This guarantees pixel preservation outside the mask, not identity preservation
inside it. Reference-conditioned editing is a separate model capability.

## Build the first three chapters

1. In Developer, use Custom memory policy and add a native Z-Image text-to-image
   connected starter. Set 1024×1024, seed `603219`, nine steps and guidance one.
   Describe a full-length adult brunette woman in an ivory tailored suit, a
   burgundy handbag, SoHo cast-iron storefronts, fire escapes, flowers, pedestrians
   and a yellow taxi. Connect Decode Latents to Preview Image and Run.
2. Inspect the photograph before continuing. Require a complete figure, readable
   outfit, plausible anatomy and detailed surroundings. Save the baseline.
3. **Save As a new chapter before editing a saved canvas.** Add Editorial Regions
   and branch the decoded image into it, retaining the original preview. Set Warm
   cool split to `0.8` and Exposure stops to `-0.3`; preview Directed image.
4. Save another chapter before adding a second Editorial Regions instance. Leave
   its grading neutral. Cover the whole wardrobe silhouette while keeping the
   face outside the edit region. Do not cut old sleeves/hands into the new garment
   blindly: that caused ghost-arm artifacts in a rejected trial. Preview the mask
   rather than assuming its coordinates fit every generated image. The selected
   chapter03 polygon is `[[0.38,0.282],[0.66,0.282],[0.73,0.6],[0.73,0.98],
   [0.32,0.98],[0.32,0.55]]` with operation `add`; chapter06 demonstrates subtraction.
5. For the rehearsed wardrobe edit, add a **FLUX.2 Klein native edit image**
   starter. Connect Directed image to its image-conditioning input. Use four
   steps and guidance one. Instruct it to replace only the ivory suit with an
   emerald silk evening gown, retaining the exact woman, pose, burgundy bag,
   shoes, background and lighting. Ordinary SDXL inpainting was tried but rejected
   visually for pose/background changes; stronger denoising alone was not enough.
   Klein's selected edit recipe does **not** take this custom mask: the mask is
   consumed by the protected compositor in the next step.
6. Add Protected Editorial Composite: original directed image → Protected
   original, diffusion output → Generated edit, region mask → Edit mask.
   Connect Final image to a new preview. Compare original, mask and composite.
7. Save, refresh and confirm prompts, region JSON, parameters and connections.
   Run from the restored graph. Do not regenerate earlier successful images just
   to repeat a screenshot or collect a missing receipt.

## Model switching and honest comparisons

Use models for their supported tasks. Text-to-image generation, ordinary img2img,
instruction editing and inpainting are not interchangeable guarantees. A new
creative brief must be distinguished from a model-only A/B comparison.

Before changing models, save a chapter and inspect the change review. Verify the
authored prompt and custom-node values survive **before** intentionally editing
them for the next artistic change. Do not imply that a changed prompt's effect
was caused solely by the model switch. Native stages and whole-pipeline routes
share a decoded-image boundary, not universally compatible latents/components.

Some model changes add a required port. In the native Klein → Qwen Image Edit
2511 change, Qwen's prompt encoder also needs the source image. Keep the existing
image-encoder branch and explicitly connect Directed image to the new Encode
Prompt image input. Run correctly remains blocked until that input is supplied.
Do not describe this new connection as something the previous recipe already had.

For a long editorial sequence, an explicitly loaded previous approved image can
be a checkpoint boundary. Label it as such: it prevents repeated generation and
keeps the current editing graph readable. It is not a hidden live connection to
the earlier graph. Retain the source image and its producing workflow together.

## Present the saved sequence

After importing a curated package, use **Workflows → My workflows**, search
**Fashion Demo**, and open the numbered chapters. Keep working attempts separate
from the selected presentation records.
Use **Save As** to make a presentation copy before experimenting. Model weights,
enabled custom code and retained local media are prerequisites, not embedded in
a workflow JSON export.

Stay in **Developer** mode with **Custom** memory policy for this graph-level
demonstration. Click an output preview to inspect it full-size; fit-to-canvas is
for explaining connections, not judging the photograph at thumbnail scale.

1. **01 — Establish the scene.** Show the full-length ivory suit, burgundy bag,
   yellow taxi, storefronts and street detail. Explain the native text encoding,
   denoising and decoding stages. This is the image all subsequent edits build on.
2. **02 — Add user code.** Show the Editorial Regions source and its actual image
   connection. Compare the original and warm-left/cool-right previews. Explain
   that this node handles polygon algebra and linear-light color processing,
   not merely a prompt string. Its controls work on decoded images independently
   of the generator family.
3. **03 — Add a generative second stage.** Show the neutral region node, mask
   preview, Klein reference edit and protected compositor. The ivory suit becomes
   an emerald silk gown. The complexity now has a concrete purpose: generated
   wardrobe replacement while retaining pixels outside the edit region.
4. **04 — Checkpoint and edit a small object.** Explain the Load Image node: it
   contains the preceding reviewed result, so the whole earlier graph need not
   rerun. Show the burgundy bag changing to metallic gold. The custom mask controls
   where the generated result is composited, not where Klein internally denoises.
5. **05 — Edit architecture.** The left storefront becomes a large flower arch
   and warm boutique window. Compare the subject and right street, which are
   outside this region. Use the mask preview to explain the scope.
6. **06 — Replace the vehicle.** The taxi becomes a cherry-red vintage convertible.
   Show the broad right-side polygon and the subtract polygon protecting the
   woman. A tight rectangle clipped the car in rejected trials: a meaningful mask
   must allow the new object's entire geometry, not just the old object's box.
7. **07 — Change the season.** Add golden branches and fallen maple leaves.
   The environment mask excludes the central subject. Point out that one custom
   node can compose several regions without proliferating one-off model nodes.
8. **08 — Change materials and weather.** Wet cobblestones, reflections, rain and
   mist require a global edit. The recognizable scene is retained, but global
   generative edits can change pose or framing slightly; do not claim exact pixels.
9. **09 — Switch native model families.** Compare the preceding Klein graph with
   native FLUX Kontext. In a live switch, inspect retained prompt/custom values
   first; then change the brief to cobalt twilight and amber shop windows. The
   image change is an artistic instruction, not a controlled model-only A/B test.
10. **10 — Return to Klein for neon night.** Compare blue-hour lighting with vivid
    magenta/cyan practical lights and reflections. The same image-boundary custom
    nodes remain useful after another native model change.
11. **11 — Make a deterministic editorial look.** This deliberately small graph
    needs no diffusion model: Load Image → Editorial Regions → Preview. Saturation
    zero and exposure `+0.2` create monochrome. Complexity is not the objective;
    each node should earn its place by changing or controlling the result.
12. **12 — Explain whole-pipeline compatibility.** Change the editing operation to
    Qwen Image 2.1. Its internal pipeline is not split into interchangeable native
    modular stages. The source image, custom region node, compositor and previews
    still connect at image boundaries. The brief restores color and changes the
    gown to sapphire blue. Opening a workflow is never consent to install a runtime.
13. **13 — Return to a native graph.** Switch back to Klein and change night into
    peach/rose-gold sunrise. Save, refresh, reopen an earlier chapter and compare
    its own retained output. Show prompts, custom parameters and connections, not
    just the last image in the browser cache.

During a presentation, use retained outputs to keep the narrative moving and run
one selected edit live. Loading another model takes time and can need runtime
activation/restart; do not promise instantaneous switching. Each chapter's full
prompt and exact mask are in its saved nodes. Mask coordinates fit this particular
photograph: changing the seed or composition requires reviewing them again.

## Local rehearsal tooling

### Selected settings

The selected images are 1024×1024 with fixed seed `603219`, batch one, Custom
memory policy, BF16 and model-CPU offload. Model defaults are not interchangeable.

| Stage | Route | Steps / guidance |
| --- | --- | --- |
| Initial generation | Z-Image Turbo native modular | 9 / 1 |
| Wardrobe and most semantic edits | FLUX.2 Klein 4B native modular edit | 4 / 1 |
| Blue hour | FLUX.1 Kontext dev native modular edit | 20 / 2.5 |
| Monochrome | Custom CPU image processing only | No denoising |
| Whole-pipeline color edit | Qwen Image 2.1 whole-pipeline edit | 20 / 1 |

Measure cold model loading separately from cached execution and browser authoring.
Activating a different optional runtime may require a backend restart; preflight
the chosen chapters before presenting. Linux shared-memory results are not
Windows GPU-memory qualification.

### Tools

The client repository provides these opt-in tools:

- `scripts/run-fashion-demo.mjs <cases.json>` with `MODIFF_FASHION_DEMO=1` submits
  chapters serially through Playwright UI gestures. It records failures, skips a
  busy backend, retains completed attempts and bounds each process. It never
  retries an unchanged failure automatically. Dependent chapters require a visual
  review bound to the exact preceding PNG hash; independent cases can continue.
- `scripts/prepare-fashion-demo.mjs <reviewed-chapters.json> <output-directory>`
  prepares reviewed local exports and an image gallery without model inference.
  Only preview output fields are bound to exact-run durable media; executable
  inputs and graph edges remain unchanged.
- `fashion-demo-checkpoints.spec.ts` imports the reviewed exports on the deployed
  frontend and checks reopening after refresh, input/edge retention and each
  displayed image's durable URL and hash. This is a targeted demo check, not a full
  UI suite or qualification of every model in the registry.
- `scripts/export-fashion-demo.mjs <manifest> <backend-data> <new-output>` creates
  a separate local transfer folder containing workflows and hash-named images.
  It rejects traversal, escaping symlinks, transient cache URLs and machine-local
  paths. It neither publishes media nor transfers model weights or code approvals.

The local ledger records accepted outputs, rejected attempts and actual timings.
Do not confuse completed inference with a visually acceptable result. Qwen Image
Edit 2511 was tried at 1024px but cancelled for impractical demo latency; it is not
a completed visual qualification of that model. It is distinct from the Qwen
Image 2.1 whole-pipeline chapter.
