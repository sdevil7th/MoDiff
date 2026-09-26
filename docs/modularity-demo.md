# Image modularity demo

> **Historical technical rehearsal, not approved visual demo content.** The robot
> sequence was rejected for weak imagery and insufficient changes between chapters.
> Use the [SoHo fashion editorial replacement](fashion-editorial-demo.md) for the
> revised creative direction. The technical instructions below remain useful as
> recorded compatibility evidence, not a recommendation to present these images.

This walkthrough uses ordinary editable nodes, not a hidden template or a new
executor. All ten choices below have completed real frontend-submitted generation
with the custom image/mask node and connected SDXL refinement at 512×512. Prompt,
custom/refiner values, connections and refresh persistence were checked. This
qualifies the demonstrated sequence, not every possible model/node combination.

## Preparation

On the rehearsed machine, open **My workflows** and search **Demo Ready**.
There are thirteen separate checkpoints: 01 baseline, 02 custom image/mask,
03 connected refinement, 04–11 native models/variants, 12 whole-pipeline Qwen 2.1,
and 13 the native return. These are editable workflows, not prerecorded runs.
Save a presentation copy before editing a checkpoint.

Use Developer mode and a fresh workflow. The top-bar **Custom** memory policy
keeps the explicitly selected loader/offload settings. Check the required model
files in Model Manager before presenting. Clear finished **Session activity**
notifications if they cover connectors; this only dismisses notifications, not
saved outputs. **Arrange graph** fits the graph after inserting nodes.

In **Nodes → Custom nodes**, stage `examples/custom_nodes/LightPaletteDirector`
under a unique module name, review the source, check consent and **Enable code**.
Saved workflows do not grant custom-code execution permission on another machine.

## 1. Build and run native text to image

1. In Nodes, choose `ZImageModularPipeline`, task **text to image**.
2. **Preview connected starter → Add starter to canvas** adds Load Models,
   Encode Prompt, Denoise and Decode Latents with their typed connections.
3. Add **Preview Image** and connect Decode Latents **Images → Image**.
4. Set 512×512, eight steps, guidance `1`, fixed seed `271828`. Use this prompt:

   > Studio product photograph of a small retro robot, cream enamel and teal metal,
   > two round eyes, on a peach pedestal, warm side lighting, full body, simple
   > backdrop, no text.

5. Run, then Save As `Modularity Demo - 01 - Native Z-Image`.

To retain an earlier chapter, **Save As the next chapter before editing**.
Editing an already-saved tab may autosave into that record. Separate imported
`Demo Ready` checkpoints are available on the rehearsed machine; duplicate one
before a presentation so the originals remain useful fallback checkpoints.

Explain that model loading, prompt encoding, denoising and decoding are independently
visible stages. The seed is fixed to make subsequent changes easier to compare.

## 2. Add meaningful custom Python

1. Add **Light & Palette Director** from the custom module in Nodes.
2. Branch Decode Latents **Images → Image** on the custom node. Keep the original
   preview connected as the before image.
3. Add two previews: **Directed image → Image** and **Soft mask → Image**.
4. Set Palette strength to `0.80`. Show Region X/Y, width/height, feather,
   shadow/midtone/highlight colors and light angle/intensity.
5. Run and compare the original, directed image and mask. Save a second checkpoint.

The custom node computes a soft elliptical mask, luminance-dependent palette and
directional gradient while preserving source detail. It runs on CPU using the
existing Pillow/NumPy dependencies. It is not a semantic segmenter or a physical
relighting model. Its decoded-image boundary is independent of the generator.

## 3. Connect custom results to diffusion refinement

1. Add a second connected starter: `StableDiffusionXLModularPipeline`, **inpaint**.
   Do not replace the first generator.
2. Connect custom **Directed image → Image** and **Soft mask → Mask Image** on
   the new Image Encode node. White mask regions are the editable region.
3. Connect the second Decode Latents to a final Preview Image.
4. Set the refiner prompt, 512×512, 20 steps, strength `0.5` and CFG 5;
   use fixed seed `271828`. Keep Image Encode and Denoise geometry in sync;
   the new SDXL starter shares those geometry controls.
5. Run and compare the three stages. Save, refresh, inspect all values and wires,
   then run again.

This makes the custom code influence a subsequent model, rather than merely
filtering the final output. Refinement strength and palette strength are different
controls. A soft input mask is processed according to the selected pipeline's mask
semantics; it does not promise arbitrary soft blending inside diffusion.

## 4. Change only the first generator

Use **Choose model** on the first Load Models node, inspect the change review,
then apply it. Leave the custom node and the separate SDXL refiner untouched.
First change `teal metal` to `cobalt blue metal` in the generator prompt. After
each switch, show that this edit survives without retyping it. Keep custom
Palette strength `0.80` and the refiner at 20 steps/CFG 5 throughout.

| Chapter | Model | Rehearsed execution |
| --- | --- | --- |
| Main | Z-Image Turbo | Editable native stages; 8 steps, guidance 1 |
| Main | SDXL Base | Editable native stages; 25 steps, CFG 5 |
| Main | FLUX.2 Klein 4B distilled | Editable native stages; 4 steps, guidance 1 |
| Extended | FLUX.1 dev | Editable native stages; 28 steps, guidance 3.5 |
| Extended | Qwen-Image-2512 | Editable native stages; 28 steps, CFG 4 |
| Variants | SDXL PAG | 25 steps/CFG 5, native Guider + explicit Layers, same SDXL checkpoint |
| Variants | SDXL Turbo | Native stages; 512px, one step, CFG 0 |
| Variants | FLUX.1 Schnell | Native stages; four steps, unused guidance hidden/fixed at 0 |
| Variants | FLUX.1 Krea dev | Native stages; 28 steps, guidance 3.5 |
| Compatibility | Qwen-Image 2.1 | Whole pipeline; 40 steps, guidance 1, same custom image boundary |

For PAG, expand **Other implementations** and select `sdxl-pag:modular`.
The repository is shared with SDXL Base, but the Guider must read
`PerturbedAttentionGuidance`; showing the same checkpoint alone does not prove PAG.
For Turbo, set CFG on the connected Guider, not an unused Denoise scalar.

Before changing model-specific steps/guidance, show that the authored prompt,
custom settings and external connections remain. Verify **Editable nodes** versus
**Whole pipeline** in the picker; do not silently substitute a standard route for
a native demonstration. Model-specific components/latents are not interchangeable.
Unsupported settings are retained inactive for restoration, not falsely active.

Save each chapter, refresh, run and return to Z-Image. Larger models can take
longer to load; a compatible graph is not a guarantee of adequate memory.
The one-step Turbo example is a speed/settings demonstration; its rehearsed
output is visibly lower-fidelity than the main models, not quality-equivalent.

## 5. Whole-pipeline compatibility and runtime changes

Qwen 2.1 replaces the generator's internal stages with its supported whole-pipeline
operation. Its decoded image still feeds the same custom node. This demonstrates
portability at the image boundary, not native Modular Diffusers support for Qwen 2.1.

The reviewed Transformers 5.17 + PEFT runtime explicitly satisfies the earlier
5.14.1 and exact-main runtime contracts. On a host with that validated environment
already active, these model changes do not need a Python restart. This does not
imply compatibility with optional quantization/media packages or arbitrary versions.

If Setup says another runtime is required, save the workflow, finish or stop active
and queued runs, explicitly install/activate the reviewed environment, and wait for
reconnection. The supervised backend restarts its worker; the UI polls readiness.
An unsupervised launch requires a manual restart. Installation is never implicit,
and failed activation must not silently run the wrong runtime or replay a generation.

## 6. Optional code reload

Edit the installed custom module's `main.py`: replace its smoothstep mask line
with `mask = mask ** 2`. In Custom nodes, **Review reload**, inspect the changed
hash, consent and enable. Keep the same ports. Run the existing graph to show
that changed code affects the mask and downstream refinement without rewiring.
See the [custom example](../examples/custom_nodes/LightPaletteDirector/README.md).

This exact edit was rehearsed on the deployed frontend: explicit code review and
consent, page refresh, then the complete graph ran successfully. The generator
image was byte-identical to the baseline while the custom image changed; the
refiner also completed and the browser reported no uncaught errors. The installed
demo copy is restored to the original smoothstep version after rehearsal.

## Presenter checklist

- Start the app and confirm models are downloaded before the presentation.
- Use **Demo Ready - 01** or a fresh workflow for the build-up; keep checkpoint
  **03** available if you want to jump straight to the connected custom/refiner
  graph. Zoom into the stage you are explaining; Arrange fits the whole graph.
- Change the prompt once, then point out its retention before adjusting each
  model's steps/guidance. Show custom strength and the refiner staying unchanged.
- Use **12** to explain whole-pipeline compatibility, then **13** for the native
  return. Do not describe model-specific latents/components as universal.
- Save, refresh, inspect values and wires, and Run. Model loading and generation
  take real time; the supported switch itself does not promise instant inference.
- For the code edit, modify the installed module, not the example source file;
  review and consent again. Finish by restoring and reapproving the original.

## Rehearsal tooling

The sibling client contains `tests/e2e/live-backend/modularity-demo.spec.ts`.
It uses real frontend authoring gestures and saves per-case workflow JSON, run
receipts, screenshots and images. It requires `MODIFF_RUN_MODULARITY_DEMO=1`,
uses bounded generation waits, and records independent model failures. It does
not run the full UI suite or automatically retry a failed model.

From the sibling client checkout, with the backend running and model files ready:

```bash
MODIFF_RUN_MODULARITY_DEMO=1 node scripts/run-modularity-demo.mjs
```

Each case records its failure and proceeds to the next. Generation waits are
bounded at ten minutes; the runner also has an outer deadline. Automatic worker
recovery is limited to the local supervised demo backend, the runner's own
active task, and an otherwise empty queue. It must not stop another user's run.
Retain the output directory: `MODIFF_DEMO_RESUME_WORKFLOW`,
`MODIFF_DEMO_SKIP_CASES` and `MODIFF_DEMO_REUSE_GENERATIONS` allow targeted
continuation instead of replaying successful generations. Reused receipts are
explicitly labelled; authoring-only success is not image-generation proof.
