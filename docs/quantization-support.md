# Quantization Support Matrix

This guide distinguishes code paths, model discovery, app delivery, and
qualification. A quantization method or model is **supported** only when MoDiff
can identify an exact model/revision, obtain the model and required runtime
through an explicit app flow, load it through an exact execution profile, and
reject incompatible hardware before Run. A source adapter, dropdown entry, Hub
catalog record, or upstream documentation link alone is not a support claim.

## User-facing support tiers

| Tier                | Meaning                                                                                                                                                          |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Auto-qualified      | The app can select and download the exact artifact, its runtime is app-delivered, and MoDiff has reviewed runtime evidence.                                      |
| App-admitted        | The app can download the exact artifact and an exact execution profile admits it, but model/platform qualification is incomplete.                                |
| Expert contract     | An exact model/mode profile exposes the method and the selected application profile supplies its dependency. Each model/hardware recipe still needs live proof.  |
| Documented research | The artifact or upstream backend is inventoried, but the app does not yet provide a complete runnable loader/runtime path. It must not be presented as runnable. |

The app downloads Hub model snapshots only after an explicit Install action. A
model selection does not install Python packages. Base accelerator profiles and
reviewed optional runtimes are installed from Setup with consent; package-only
optimization entries remain disabled when their immutable dependency lock is
not available. The Setup > Optimizations panel links to the relevant upstream
documentation, but following a `pip install` example outside MoDiff does not
turn that environment into a supported MoDiff runtime.

## Current end-to-end model paths

| Model/artifact                                                                                      | Quantization            | Tier                                            | App path and boundary                                                                                                                                                                                 |
| --------------------------------------------------------------------------------------------------- | ----------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `unsloth/Qwen-Image-2512-unsloth-bnb-4bit`                                                          | pre-quantized BnB 4-bit | Auto-qualified on reviewed NVIDIA CUDA profiles | Studio Auto or Model Manager can request the immutable Hub snapshot. NVIDIA setup supplies `bitsandbytes`; the exact Qwen text-to-image profile admits this fallback. Other platforms remain blocked. |
| `black-forest-labs/FLUX.1-dev-FP8`                                                                  | native FP8 artifact     | App-admitted on declared CUDA/ROCm paths        | The exact FLUX Dev profiles admit the official immutable alternate and Auto can request it as the lower-memory artifact. Each platform/model recipe still requires a current live resource receipt.   |
| `black-forest-labs/FLUX.1-Kontext-dev-NVFP4`                                                        | native NVFP4 artifact   | App-admitted on NVIDIA Blackwell Linux          | The exact Kontext edit/inpaint profiles admit the official immutable alternate. Preflight requires CUDA compute capability 10.0 or newer. It is not a generic FP4 fallback.                           |
| official Qwen Image and Qwen Image Edit-family repositories                                         | on-load BnB 4-bit       | Expert contract on NVIDIA CUDA                  | Exact Qwen profiles expose only `bnb_4bit`, with reviewed component selection and offload policy. It is not Auto-defaulted where prior kernel failures are known.                                     |
| official FLUX Schnell, Dev, Krea, Depth, Canny, Redux, Kontext, Fill, and FLUX.2 Klein repositories | on-load BnB 4/8-bit     | Expert contract on NVIDIA CUDA                  | The NVIDIA application profile installs `bitsandbytes==0.50.0`. Exact model/mode profiles bound the choices; support is not inferred for arbitrary Diffusers pipelines.                               |
| `city96/FLUX.1-schnell-gguf` / `flux1-schnell-Q4_0.gguf`                                           | GGUF Q4_0              | Expert exact-file contract                      | The component node downloads the pinned 6,770,707,360-byte file, verifies SHA-256, loads `FluxTransformer2DModel.from_single_file`, and assembles only the pinned FLUX.1-schnell bfloat16 base.          |
| `HuggingFaceTB/SmolLM2-135M-Instruct`                                                              | on-load BnB NF4         | Expert contract on NVIDIA CUDA                  | The exact text profile exposes `bnb_4bit`; the loader requires the pinned commit, bfloat16 compute, CUDA, NF4, double quantization, and the NVIDIA base profile's `bitsandbytes==0.50.0`.                 |

No other quantized model currently meets the complete end-to-end support
definition. In particular, the official FLUX.2 Klein FP8 catalog entry is not
yet connected to an exact compatible-repository profile, and the community
Qwen Image Edit 4-bit artifact is not Auto-qualified.

The only quantized Transformers-only path is the exact SmolLM2 BnB contract
above. Image/video-to-text, speech, and any-to-any loaders do not accept
quantization controls. `bnb_8bit`, GPTQ, AWQ, GGUF, and arbitrary
`quantization_config` input remain unavailable for Transformers nodes.

## Runtime quantization backends

| Backend/mode                     | Implemented loader API                                                                                                     | App-delivered dependency                                                             | Studio admission                           | Current disposition                                                               |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------ | --------------------------------------------------------------------------------- |
| `bnb_4bit`                       | Yes, including separate Diffusers and Transformers component configs                                                       | Yes in the NVIDIA CUDA application profile                                           | Exact Qwen and FLUX profiles               | Supported only at the tiers listed above.                                         |
| `bnb_8bit`                       | Yes                                                                                                                        | Yes in the NVIDIA CUDA application profile                                           | Exact FLUX profiles                        | Expert contract; no blanket model/platform claim.                                 |
| `quanto_float8`                  | Yes                                                                                                                        | Yes; exact Transformers/PEFT/Quanto overlay on qualified Linux x86-64                | Exact FLUX profiles when that overlay is active | Expert contract; isolated ROCm float8 qualification passed, while each full model recipe still needs live proof. |
| `quanto_int8`                    | Backend generic loader only                                                                                                | No                                                                                   | Not admitted by Studio profiles            | Documented research.                                                              |
| `torchao_float8`                 | Yes                                                                                                                        | No; the legacy `torchao` package entry has no reviewed immutable install lock        | Hidden as unavailable at runtime            | Not end-to-end supported.                                                         |
| `torchao_int8_weight_only`       | Backend generic loader only                                                                                                | No                                                                                   | Not admitted by Studio profiles            | Documented research.                                                              |
| `torchao_mxfp8`, `torchao_nvfp4` | Artifact-creation helper only                                                                                              | No supported app environment                                                         | Not admitted                               | Research/qualification tooling, not a user runtime feature.                       |
| GGUF                             | Exact FLUX.1-schnell Q4_0 component loading and base-pipeline assembly                                                      | Supplied by the reviewed base Diffusers runtime                                      | Exact component-node contract               | Expert exact-file path; other GGUF catalog entries remain research only.          |

Diffusers' pipeline quantization needs component-aware configuration. A
Transformers text encoder requires `transformers.BitsAndBytesConfig`, while a
Diffusers transformer requires `diffusers.BitsAndBytesConfig`. GGUF is also not
a pipeline repository substitution: upstream loads a quantized component from
a single file and assembles it with the remaining pipeline components. Treating
either case as a generic repository swap is unsafe.

## Quantized artifact inventory

`data/model-artifact-catalog.json` currently contains 22 quantized artifact
records: two Diffusers-BnB, four FP8, fifteen GGUF, and one NVFP4. The catalog is
an immutable discovery and review ledger, not an execution allowlist.

| Model type                         | Artifact                                     | Format              | Review status                                          |
| ---------------------------------- | -------------------------------------------- | ------------------- | ------------------------------------------------------ |
| `ZImageModularPipeline`            | `T5B/Z-Image-Turbo-FP8`                      | FP8                 | Community research                                     |
| `QwenImageModularPipeline`         | `unsloth/Qwen-Image-2512-unsloth-bnb-4bit`   | Diffusers-BnB 4-bit | MoDiff-qualified path                                  |
| `QwenImageEditModularPipeline`     | `ovedrive/qwen-image-edit-4bit`              | Diffusers-BnB 4-bit | Community option; not Auto-qualified                   |
| `QwenImageEditPlusModularPipeline` | `unsloth/Qwen-Image-Edit-2511-GGUF`          | GGUF 4-bit          | Popular unverified research                            |
| `QwenImageEditPlusModularPipeline` | `1038lab/Qwen-Image-Edit-2511-FP8`           | FP8                 | Popular unverified research                            |
| `QwenImageLayeredModularPipeline`  | `unsloth/Qwen-Image-Layered-GGUF`            | GGUF 4-bit          | Popular unverified research                            |
| `WanVACEPipeline`                  | `samuelchristlie/Wan2.1-VACE-1.3B-GGUF`      | GGUF 4-bit          | Community research                                     |
| `WanVideoPipeline`                 | `samuelchristlie/Wan2.1-T2V-1.3B-GGUF`       | GGUF 4-bit          | Popular unverified research                            |
| `WanImageToVideoPipeline`          | `QuantStack/Wan2.2-I2V-A14B-GGUF`            | GGUF 4-bit          | Popular unverified research                            |
| `WanTI2VPipeline`                  | `QuantStack/Wan2.2-TI2V-5B-GGUF`             | GGUF 4-bit          | Popular unverified research                            |
| `LTXVideoPipeline`                 | `QuantStack/LTXV-13B-0.9.8-distilled-GGUF`   | GGUF 4-bit          | Community research                                     |
| `AceStepAudioPipeline`             | `Serveurperso/ACE-Step-1.5-GGUF`             | GGUF 5-bit          | Popular unverified research                            |
| `FluxSchnellPipeline`              | `city96/FLUX.1-schnell-gguf`                 | GGUF 4-bit          | Popular unverified research                            |
| `FluxDevPipeline`                  | `black-forest-labs/FLUX.1-dev-FP8`           | FP8                 | Official app-admitted alternate                        |
| `FluxKreaPipeline`                 | `QuantStack/FLUX.1-Krea-dev-GGUF`            | GGUF 4-bit          | Popular unverified research                            |
| `FluxKontextPipeline`              | `black-forest-labs/FLUX.1-Kontext-dev-NVFP4` | NVFP4 4-bit         | Official app-admitted Blackwell alternate              |
| `FluxFillPipeline`                 | `YarvixPA/FLUX.1-Fill-dev-GGUF`              | GGUF 4-bit          | Popular unverified research                            |
| `FluxDepthPipeline`                | `SporkySporkness/FLUX.1-Depth-dev-GGUF`      | GGUF 4-bit          | Community research                                     |
| `FluxCannyPipeline`                | `SporkySporkness/FLUX.1-Canny-dev-GGUF`      | GGUF 4-bit          | Community research                                     |
| `FluxReduxPipeline`                | `second-state/FLUX.1-Redux-dev-GGUF`         | GGUF 4-bit          | Community research                                     |
| `Flux2KleinPipeline`               | `black-forest-labs/FLUX.2-klein-4b-fp8`      | FP8                 | Official catalog entry; exact loader admission pending |
| `Flux2KleinPipeline`               | `unsloth/FLUX.2-klein-4B-GGUF`               | GGUF 4-bit          | Popular unverified research                            |

## Upstream coverage and function documentation

MoDiff does not claim support for every model on the Hugging Face Hub. The Hub
is open-ended and many repositories require custom code. The bounded product
scope is:

- every top-level `*Pipeline` export in the exact pinned Diffusers source is
  classified in `data/upstream-coverage.v1.json`;
- each finite Transformers product semantic is classified there separately;
- every executable model/mode pair is bound to an exact execution specification
  and generic node/action contract; and
- the live `/nodes`, `/studio/model-capabilities`, and
  `/studio/execution-specifications` responses document the callable fields,
  types, modes, loaders, artifacts, and runtime requirements used by the app.

Internal helper functions, aliases, training APIs, and arbitrary Hub custom
code are not separate product features. They should be linked to upstream API
documentation when used, but they must not be counted as supported merely
because they are importable. The coverage ledger's `executable` status proves
source/graph admission only; live output, quality, license, hardware, Auto,
Gallery, and release qualification remain separate evidence.

## Closure work

Remaining expansion work:

1. Deliver an immutable TorchAO runtime before making TorchAO selectable.
2. Add model-level live load/run/resource evidence before promoting the exact
   Quanto, GGUF, or Transformers BnB contracts beyond Expert.
3. Connect only reviewed pre-quantized repositories to exact compatible model
   profiles; catalog popularity must never create execution compatibility.
4. Record live load/run/resource receipts per model, mode, backend, and
   quantization recipe before promoting an Expert contract to Auto-qualified.
5. Keep the upstream coverage ledger and public API schemas synchronized with
   the exact pinned Diffusers and Transformers revisions.

Official upstream references:

- [Diffusers quantization overview](https://huggingface.co/docs/diffusers/main/en/quantization/overview)
- [Diffusers bitsandbytes](https://huggingface.co/docs/diffusers/main/en/quantization/bitsandbytes)
- [Diffusers GGUF](https://huggingface.co/docs/diffusers/main/en/quantization/gguf)
- [Diffusers TorchAO](https://huggingface.co/docs/diffusers/main/en/quantization/torchao)
- [Diffusers Quanto](https://huggingface.co/docs/diffusers/main/en/quantization/quanto)
- [Transformers quantization overview](https://huggingface.co/docs/transformers/main/en/quantization/overview)
