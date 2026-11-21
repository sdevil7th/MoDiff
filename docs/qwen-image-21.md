# Qwen-Image 2.1 in generic image workflows

The standard Diffusers adapter uses the existing **Load Pipeline**, **Generate
Image**, **Edit Image** and **Decode Image Latents** nodes. Choose Qwen Image 2.1
in Developer → Workflows, then choose text-to-image, image edit or multiple
reference images. The graph remains editable and can be saved or exported
through the existing workflow and service-package paths.

The model revision is `790c92633540aa0cb11d9abf19eb46d861714758` in
[`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1).
Weights use the publisher's [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/LICENSE),
which limits use to non-commercial research/evaluation unless separately
licensed. Model weights and generated Gallery assets are not distributed with
this integration.

## Runtime and current qualification

The historical source review inspected
[`fbf49e7f35857f76bc57b177e26f12b03687c668`](https://github.com/huggingface/diffusers/commit/fbf49e7f35857f76bc57b177e26f12b03687c668).
Ordinary setup pins the reviewed stable release `diffusers==0.43.0`, with
artifact hashes recorded in `uv.lock`; the historical source revision does not identify the
installed wheel. Modular Diffusers is part of the same package. Transformers
and PEFT come with ordinary backend setup, with compatible versions recorded
in `uv.lock`.
Qwen 2.1 needs no separate core runtime installation or activation. Older exact
optional profiles describe historical qualification and are not prerequisites
for a verified base environment. GGUF and other specialized optional extras
retain their own compatibility requirements.

Generic adapter contracts, a tiny real CPU denoiser/VAE test, and full-weight
1024×1024/40-step text generation and two-reference editing pass on the reviewed
Linux AMD runtime. Tests include native KV on/off, prompt/seed edits, reopening
and UI/service agreement. The ROCm vision encoder uses eager attention to avoid
non-finite SDPA embeddings; text and denoiser attention remain unchanged.
See the [image demo checkpoint](image-demo.md) for measured timings, exact test
scope and source revisions. Full release acceptance and reference-replacement
coverage remain open; this does not establish Windows 16 GB VRAM / 32 GB RAM viability.

MoDiff uses the standard Diffusers pipeline for these routes. The installed
library also ships Qwen Image 2.1 Modular pipelines. Separate Modular graph
integration remains unqualified. The generic decoder accepts the standard
pipeline's emitted latents.

## Turbo checkpoint

Choose **Qwen Image 2.1 Turbo** for the same text generation, image edit and
ordered-reference edit workflows. MoDiff installs the 18 selected files at
`fc13ba1986fcb905854dc8de2f52eee395f49b5b` through Model Manager. This
checkpoint has its own [model terms](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/LICENSE)
for non-commercial research and evaluation. The model terms acknowledgement
is scoped to this repository and revision.

The native Diffusers pipeline reads the checkpoint's eight-step sampling
schedule with guidance 1.0. The image stages hide steps, guidance, custom
sigmas and the negative prompt for this profile. Prompt, seed, dimensions,
reference order and attention-context reuse remain available. Switch between
Base and Turbo through Model Setup to load the matching checkpoint and
task controls. Base retains its 40-step default and adjustable guidance.
[Official Turbo model and generation example](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)

## Reuse attention context

**Reuse attention context** forwards the native `use_kv_cache` boolean. It is
enabled by default for this pipeline and appears only on compatible generic
nodes. During a generation, the first denoising step prefills prompt and
reference context; later steps reuse the fixed context. The upstream pipeline
owns a fresh cache for each invocation, including separate positive and
negative guidance caches when applicable.

This differs from MoDiff reusing loaded models, prompt embeddings or a completed
node output between graph runs. The execution receipt records the actual cache
flag passed to Diffusers. It does not estimate cache hits or claim that one
request reused another request's KV state.

Caching trades memory for less repeated computation. Cache on/off may produce
different reduced-precision images; fix the flag when comparing repeated
samples. The publisher's A100 timing is not a benchmark for other hardware.
See the [native pipeline contract](https://github.com/huggingface/diffusers/blob/fbf49e7f35857f76bc57b177e26f12b03687c668/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py).

## Images, references and latents

The generation and editing routes share one pipeline. Multiple-reference edit
accepts up to ten references within the generic cumulative input-pixel budget.
RGB and RGBA media remain intact through the adapter. New workflows use the
publisher's 2048×2048, 40-step recommendation with guidance 1.0. Dimensions use
32-pixel increments, with a 4096-pixel side and 5-megapixel output ceiling;
these bounds also permit the publisher's wide 2K presets.

Choose latent output to connect **Decode Image Latents** explicitly. Use the
same loaded pipeline and original dimensions. The decoder applies that VAE's
native mean/standard-deviation normalization and preserves its alpha channel;
latents from another model are not interchangeable merely because the port is
a tensor. Controls absent from the native pipeline, such as edit strength or a
maximum sequence-length override, remain hidden.
