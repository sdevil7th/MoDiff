# Large media integrations

MoDiff's ordinary dependency contract pins the official Hugging Face Diffusers
0.43.0 wheel. Its wheel hash is recorded in `uv.lock`. Historical upstream catalog
source and optional provider revisions retain their own identities; installing
the newer release does not qualify a model, device or old provider automatically.

## Kandinsky 6 Pro

The shared `DiffusersVideo.LoadPipeline`, `Generate` and `GenerateVideoAudio`
nodes select the small backend adapter for the official
`Kandinsky6TI2VAPipeline`. No publisher Python or alternate graph executor is
loaded from the model repository.

| Input or operation | Contract |
| --- | --- |
| Text generation | One prompt and optional negative prompt |
| First-frame conditioning | Exactly one reference image; no mask or source-video route |
| Video only | `Generate` requests `sample_audio=False` |
| Synchronized video and audio | `GenerateVideoAudio` preserves both native streams and reads `pipeline.audio_sample_rate` |
| Prompt expansion | The generic Expand Prompt control maps to the official local `expand_prompts` argument; disabled by default |
| Sampling | Seed, steps, joint guidance, dimensions, frame count and frame rate reach the official call |
| Geometry | Positive dimensions divisible by 16; frame count follows the selected VAE's causal temporal grid |
| Decoded formats | Native PIL, NumPy and Torch results normalize to ordered PIL video frames without applying denormalization twice |
| Placement | Existing resource-managed full residency or model CPU offload; no new group/quantization qualification is implied |

The Full checkpoint is pinned at
`kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers@7be18aed6fbd64653c0c4872fddfe4a3e3e109f5`.
The separate Distilled checkpoint is pinned at
`kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers@80d5e6c7f646c5f512b684e2da0765618d805894`.
Publisher starting settings are 50 steps / guidance 5 for Full and 10 steps /
guidance 1 for Distilled. Saved authored values are preserved when selecting a
different checkpoint; the app must not silently substitute one checkpoint or
rewrite an existing recipe. Negative prompting is governed by the selected
guidance, as in the official pipeline.

The official API has one shared prompt and guidance control. Separate audio CFG
is not exposed. Precomputed prompt embeddings, audio latents, custom timesteps/sigmas, batching, the independent
super-resolution pipeline and adapter training/loading are not admitted by this
initial facade. These remain explicit integration work, even though the SDK may
provide some of those interfaces. Native prompt expansion occurs inside the
pipeline; its returned output does not expose the expanded prompt, so records
must distinguish the authored prompt and expansion flag from an unavailable
expanded-text receipt.

Kandinsky exposes `text_to_video` and `image_to_video` for video-only output, and
`text_to_video_with_audio` and `image_to_video_with_audio` for synchronized streams.
Each fresh draft selects one matching generation operation and output stage;
switching a video-only model does not silently discard newly generated audio.
Existing LTX AV drafts select the single operation declared by their exact source
recipe while preserving both legacy operation contracts. Unsupported inherited input fields are
hidden for this adapter and rejected before inference if supplied by an older or
manually edited graph.

Static call-surface and no-weight adapter tests establish argument translation
and output handling. They are not pretrained output review, hardware
qualification, audio listening or template publication approval. New selections
remain Expert-only until those separate checks pass.

## Oversized Hub files

Model Manager keeps normal Hub downloads on bounded HTTP. Hugging Face's HTTP
transport rejects a single file larger than 50 GB, including Kandinsky Pro
Full's 69,734,820,096-byte transformer. Those files use the official Hub Xet
transport in a disposable process with a 30-minute transfer window. The same
immutable repository revision, finite file plan, managed cache validation and
download lease apply. Credentials travel over a private input pipe, not process
arguments or temporary files. A timeout terminates that worker and preserves
partials for a later resume. This transfer timeout is separate from model
generation scheduling. Repair still coordinates with normal downloads through
the existing exclusive repair gate.

## HunyuanVideo 1.5 720p

The full 720p T2V and I2V checkpoints use the existing Model Setup, Encode Inputs,
Generate and Video Output stages. I2V keeps its separate source-image input and
the VAE/SigLIP encoding steps inside Encode Inputs. Select an exact reviewed
checkpoint in Model Setup; no model-specific frontend node is required.

The T2V revision is `f4dbc4a1efa4ac8ea56680cdf79d9f455105e814` in
`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v`; I2V is
`dcf088ef4e420dd54bc25930ee810bb6613dfcb5` in
`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v`. Fresh full recipes
start at 1280×720, 121 frames, 24 FPS, 50 steps and CFG 6. The existing 480p
step-distilled I2V recipe retains 12 steps and fixed CFG 1. The connected immutable
checkpoint determines whether ordinary I2V guidance is allowed. Image dimensions
belong to the VAE input stage and are not injected into the downstream denoiser.
Saved authored settings are not rewritten.

Explicit Modular denoise stages share the same temporary progress bridge into
the existing task queue. It instruments the executing loop tree, reports actual
completed steps and estimates, respects Stop after a model step, and restores
upstream progress methods after success or failure. Loading shard counts are
not sampling steps. The client must honor explicit null telemetry when the
backend moves to a different phase; omitted legacy fields remain unchanged.

Weight placement and decode working memory are separate requirements. Hunyuan's
native VAE defaults to untiled decoding; full GPU residency is not proof that a
720p clip's temporary decode allocation fits. The existing explicit VAE Tiling
policy uses the official VAE hooks. Batch slicing alone does not reduce memory
for a single video. Failed ordinary graph attempts currently release all runtime
caches. Eligible HunyuanVideo 1.5 sampling automatically retains bounded latent
tensors and provenance before decode. The app currently saves this material for
inspection and future recovery. Retrying a failed decode does not yet restore
the saved tensor automatically.
Do not promise decode-only recovery after that cleanup or a worker restart.

Isolated official stages retain the connected transformer's dtype and target
size, and the connected VAE's latent geometry, through explicit component
dependencies. Splitting the visual stages must not substitute fallback geometry
or drop a component read by the official stage implementation.

The exact file selections are safetensors-only. Base PyTorch attention is
available; browsing or running does not fetch optional Hub attention kernels.
Full GPU residency and model CPU offload are declared placement choices, not a
hardware qualification. Auto, Gallery, live-proof and publication eligibility
remain separate. The revision-bound Hunyuan community-license notice includes
its AUP, territory, MAU, redistribution and generated-content disclosure terms.

## Remaining suite

Hunyuan 720p native acceptance, LTX-2.5 Full, PAN 2, SANA-WM and the MiniMax/Pika/P-H3
feature audit remain integration work. An SDK class, a Hub repository or sufficient VRAM alone is not an
executable integration. Required safety packages, packed-weight formats,
publisher terms and GPU kernels must each have a compatible reviewed path.

Sources: [official Diffusers release](https://github.com/huggingface/diffusers/releases/tag/v0.43.0),
[Kandinsky 6 API](https://huggingface.co/docs/diffusers/main/en/api/pipelines/kandinsky6),
and the immutable model cards linked by the catalog's repository revisions.
