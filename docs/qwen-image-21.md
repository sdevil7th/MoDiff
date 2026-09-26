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

The reviewed Diffusers source is
[`fbf49e7f35857f76bc57b177e26f12b03687c668`](https://github.com/huggingface/diffusers/commit/fbf49e7f35857f76bc57b177e26f12b03687c668).
Modular Diffusers is part of that same package. Qwen 2.1 requires the separate
Transformers 5.17.0 / PEFT 0.20.0 optional profile, with Tokenizers 0.23.1.
Browsing a workflow does not install it; use the app's explicit runtime setup
action. This new optional profile has passed isolated Linux x86-64 installation,
symbol, PEFT computation, activation and rollback qualification. Other targets
remain pending for this profile. Its exact compatibility aliases cover the reviewed
base and main Transformers profiles used by existing image/audio workflows;
GGUF, bitsandbytes and other optional extras still require their own profiles.

Generic adapter contracts, a tiny real CPU denoiser/VAE test, and full-weight
1024×1024/40-step text generation and two-reference editing pass on the reviewed
Linux AMD runtime. Tests include native KV on/off, prompt/seed edits, reopening
and UI/service agreement. The ROCm vision encoder uses eager attention to avoid
non-finite SDPA embeddings; text and denoiser attention remain unchanged.
See the [image demo checkpoint](image-demo.md) for measured timings, exact test
scope and source revisions. Full release acceptance and reference-replacement
coverage remain open; this does not establish Windows 16 GB VRAM / 32 GB RAM viability.

The reviewed upstream source has **no native Qwen 2.1 Modular pipeline**. This
integration runs its standard Diffusers pipeline. Exposing a latent output and
an explicit decoder does not imply independently composable native prompt and
denoise blocks.

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
