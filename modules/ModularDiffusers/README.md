<!-- Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff. -->

# Modular Diffusers in MoDiff

MoDiff integrates the experimental [Diffusers Modular Pipelines](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/overview) APIs with its node graph. A small set of dynamic nodes can expose different model pipelines without creating a separate hardcoded node class for every model family.

> [!WARNING]
> Modular Diffusers APIs and compatible Hub repositories are still evolving. A visible node contract is not proof that every model/revision will load or fit on the current hardware. Custom repository contracts are preview-only in this release; repository-code execution and `trust_remote_code` are fail-closed. Read [SECURITY.md](../../SECURITY.md).

## Concepts

- **Dynamic node contracts:** node fields adapt to the selected pipeline configuration.
- **Composable workflows:** model loading, prompt encoding, denoising, and decoding can remain separate or be combined into a custom block.
- **Shared components:** compatible nodes can reuse components from the package-level `ComponentsManager` instead of loading duplicate models.
- **Hub-backed contract previews:** exact cached Hub commits can provide bounded declarative metadata used to construct a node interface; custom execution remains disabled.
- **Resource controls:** loaders expose supported quantization and offload modes, subject to package, model, and hardware compatibility.

MoDiff adapts Diffusers' Mellon node-metadata helper to supply MoDiff dynamic fields and configuration names; the
derivation is recorded in `pipeline_schema.py` and `THIRD_PARTY_NOTICES.md`. Upstream Diffusers remains responsible
for model components and Modular Pipeline execution.

## Setup

Install and validate the backend from the repository root:

```bash
./install.sh --accelerator auto
./.venv/bin/python -m modiff.preflight --json --check-port 8088 --fail-on-error
./run.sh
```

On Windows, use `install.ps1`, `.venv\Scripts\python.exe`, and `run.ps1` as shown in the root quick start. The
managed installer owns the executable Torch profile; `uv sync` and `uv run` are intentionally unsupported.

Open <http://127.0.0.1:8088>. Keep the server on loopback; it has no authentication or remote-code sandbox.

Optional quantization, Nunchaku, or other acceleration paths require the matching profiles described in the [root README](../../README.md#managed-installation-profiles).

## Start with a bundled graph

Open the workflow library in the left sidebar and expand `modular_diffusers`. The repository currently includes:

- `text_to_image` — a separated text-to-image pipeline.
- `image_to_image` — prompt plus reference-image conditioning.
- `multiple_image_edit` — multiple-image editing inputs.
- `quantization` — an example with an explicit quantization configuration.

Drag a runnable built-in graph onto the canvas, inspect its selected model and required inputs, then update the graph before running. Models are not bundled with these JSON files; MoDiff may need to download them, and gated repositories may require accepted terms plus a Hugging Face read token.

[Watch the bundled workflow browser demo (MP4)](https://github.com/user-attachments/assets/a4d0604f-80ea-4470-80e6-53a73e584ca3)

## The five-node workflow

The bundled `text_to_image` graph illustrates five stages:

1. **Load Models** selects the pipeline type and exposes compatible text encoders, denoise model, VAE, scheduler, and optional image encoder components.
2. **Encode Prompt** converts prompt fields into the embeddings expected by the selected pipeline.
3. **Denoise** performs the iterative generation step using width, height, step, guidance, seed, and model-specific fields.
4. **Decode Latents** converts latent output into image output with the VAE.
5. **Preview Image** publishes the generated image to the client/cache surface.

The exact fields and defaults come from the live registry. For example, Flux, Qwen Image, Z-Image, and Wan pipelines do not share one universal guidance, prompt, or step contract. Refresh or recreate a graph when a model's dynamic definition changes.

Reviewed built-in pipeline metadata also declares any additional component that
**Load Models** must load and publish. Wan I2V currently declares its
`image_encoder` this way; the loader no longer selects that requirement from a
pipeline-class branch. The same bounded field in an untrusted custom sidecar is
descriptive only and cannot authorize custom execution.

The **Layers** node likewise derives its selectable transformer-stack paths
from reviewed built-in pipeline metadata. Both dynamic field creation and graph
execution require the connected model signal and reject an absent, unknown, or
unlisted block path. A custom sidecar cannot extend this executable allowlist.

The **Guider** node narrows its class selector from the same reviewed pipeline
metadata and validates the connected model signal again at execution. Pipelines
without an upstream Guider component expose no executable choice, and guiders
that consume layer stacks are offered only when that pipeline has a reviewed
Layers allowlist.

The **Scheduler** replacement selector follows the pinned upstream compatibility
contract. SDXL and Wan expose only scheduler classes compatible with their
expected Euler or UniPC component; flow-matching pipelines expose no legacy
replacement choices. Dynamic field refresh and execution require the connected
reviewed pipeline identity, and execution also verifies the live scheduler
component class before replacement.

The **Denoise** node also uses reviewed pipeline metadata for the narrow legacy
case where hidden `height` and `width` values must remain available alongside
image latents. Other and unknown pipelines discard those stale dimensions, and
custom sidecar metadata cannot authorize a built-in execution exception.

[Watch a separated Modular Diffusers workflow demo (MP4)](https://github.com/user-attachments/assets/4bbf74ac-404e-46bb-ae51-a84e65c25235)

Type a prompt, confirm model readiness, and use **Run**. A queued task response only confirms submission; watch Queue and WebSocket progress for completion or structured failure details.

[Watch a workflow execution demo (MP4)](https://github.com/user-attachments/assets/e563eeb0-4f9e-4a27-8304-49fd15b87550)

### Opaque state routes

Some reviewed built-in Qwen, SDXL, and Wan I2V action contracts dynamically
expose `Route State` connectors. Wan I2V uses **Image Embeddings** → **Encode
Image** → **Denoise** → **Decode Latents**, with image embeddings, image
condition latents, and denoised latents retained on their exact typed edges.
Keep the ordinary typed connections as well as the route connection. The route is a
process-local capability that binds the exact Models Loader execution,
component roles and resident processors, generator state, routed geometry,
crop/overlay state, and paired tensors;
it is not model data and cannot be serialized, copied between loader runs, or
restored from an imported workflow value. Rerun the loader and upstream action
when a route is missing or stale.

Studio switches to the native mask/overlay path only after the complete route
chain is present. A partial dynamic definition remains pending instead of
guessing a fallback topology. The generic Qwen path and the internal SDXL base
inpaint path carry masks and masked-image latents on their typed graph edges.
The internal Modular SDXL inpaint path remains unadvertised, unprofiled, and
unqualified; the separately profiled standard Diffusers inpaint adapter does
not change that boundary. Its internal VAE route may be combined with the
generic SDXL ControlNet bundle only when
the Load Model output is a current, exact `ControlNetModel` or
`ControlNetUnionModel` publication and the selected ordinary/Union variant
matches that class. Selecting Union reveals one bounded numeric control-type
index; Denoise also requires that index to exist in the resident model's
declared `num_control_type` contract. The same resident component must survive
cache validation, pipeline initialization, component installation, and the
upstream call. The internal generic **IP-Adapter Embeddings** action can add one
reviewed standard SDXL adapter to that same resident UNet and pass its exact
positive/negative embeddings to Denoise. It accepts only
`h94/IP-Adapter@018e402774aeeddd60609b4ecdb7e298259dc729` and
`sdxl_models/ip-adapter_sdxl.safetensors`, verifies the cataloged byte size and
SHA-256 from the local Hub cache, and never downloads during graph execution.
Its image encoder also loads locally from the pinned repository revision and
must match the reviewed CLIP ViT-H geometry. The process-local receipt binds the
loader execution, UNet mutation, adapter parameters/scale, encoder, processor,
Guider, source pixels, and embedding tensors through cache and Denoise
boundaries. Re-running Models Loader removes only that current owned mutation
before issuing a new loader receipt. This single-adapter path is contract-only:
it is not a public mode or template, requires the optional Transformers runtime
to have been installed explicitly, and has no live output qualification.
Multiple adapters and Multi-ControlNet remain disabled. Wan
first/last-frame topology remains unadvertised, but its official artifact is
reviewed at an immutable revision. The generic Models Loader accepts that exact
repository variant, and Image Embeddings plus Encode Image require the selected
I2V/FLF workflow to match the loader publication before initializing blocks.
The distinct FLF processor and transformer contracts are then revalidated by
the existing route-state boundary; changing only `last_image`, repository, or
revision fails closed.

## Reusing a loaded model

Compatible tasks can share components from one `Load Models` node. For example, an image-edit path can add image encoding/conditioning nodes while reusing the model components already loaded for text-to-image.

Component reuse depends on compatible pipeline contracts and current cache state. It reduces duplicate loading but does not guarantee that every model remains resident or that a new task avoids additional allocations.

[Watch a model-reuse demo (MP4)](https://github.com/user-attachments/assets/ddbc3e06-6254-4595-8209-4cfd98d3aabc)

## Dynamic Block

`Dynamic Block` previews the declarative node contract from a compatible Modular Diffusers repository. Enter a neutral repository ID and an exact 40-character commit, cache that revision through Model Manager, then inspect its sanitized generated fields. Previewing reads only the local Hub cache, performs no network fetch, and does not construct or execute the upstream pipeline.

Dynamic blocks are not arbitrary no-code plugins. Sidecars must use MoDiff's bounded declarative schema, and executable callbacks are rejected.

Execution accepts only an exact Hub snapshot whose canonical `modular_model_index.json` selects a pinned installed Diffusers pipeline/block pair, declares only the official Diffusers or Transformers component libraries, and pins every main and auxiliary component repository. MoDiff copies the reviewed sidecar and canonical index into a private content-addressed execution snapshot before constructing installed blocks. Local mutable repositories remain preview-only. Repository Python remains unavailable without a fresh task-scoped operator authorization; a workflow checkbox or checksum is never consent.

## Combining workflows

Multiple Modular Diffusers paths can coexist on one canvas and share compatible components. A second block can consume the output of the first while reusing loaders, or separate denoise/decode paths can compare schedulers and resource modes.

Only nodes connected to the submitted graph path execute, but shared component state still consumes memory. Inspect Queue, loader diagnostics, and GPU-process information when a combined graph exceeds available resources.

## Reviewed custom Hub contracts

MoDiff can inspect compatible custom contracts from an exact locally cached Hugging Face Hub commit. Preview is bounded and does not construct a pipeline, install optional libraries, or fetch from the network. Execution is available only when the repository's canonical upstream contract resolves entirely to the reviewed installed Diffusers surface:

Custom block repositories must publish MoDiff's current `modiff_pipeline_config.json` schema. The loader does not fall
back to earlier extension schemas or filenames.

1. Review the repository, owner, dependencies, license, and exact commit.
2. Enter the reviewed 40-character commit revision; moving branches and tags are rejected.
3. Keep `trust_remote_code` off. The backend rejects it before identity issuance, cache reuse, or model construction.
4. Review the canonical `modular_model_index.json`: the pipeline and blocks must match MoDiff's pinned upstream contract, component libraries are limited to official Diffusers or Transformers exports, and every auxiliary Hub repository must have an exact commit.
5. Install any required Transformers/PEFT runtime explicitly through Setup. Preview itself never requests or installs that runtime.
6. Run only after the backend revalidates the exact repository identity and creates its private content-addressed metadata snapshot. Local mutable repositories remain preview-only.

Repository Python remains disabled. Supporting it would require a fresh task-scoped operator authorization that cannot be persisted in or restored from a workflow; the current trust checkbox is not that authorization.

[Watch the historical custom prompt block demo (MP4)](https://github.com/user-attachments/assets/d68bc8c1-1b1c-478a-b94b-1e498c60a4fc). It predates the current fail-closed execution boundary and is not current qualification evidence.

## Additional nodes

### Load Models and Load Model

`Load Models` constructs the component set for a known pipeline configuration. `Load Model` can replace or provide an individual component. Hub and local sources are supported where the selected loader contract allows them.

Local folders discovered by diagnostics are not automatically runnable model packages. The loader still needs the expected Diffusers configuration, files, revision, and component type.

### Quantization Config

Quantization nodes create explicit configuration objects consumed by compatible loaders. Available modes depend on the installed `quantization` extra, platform markers, PyTorch/CUDA versions, model architecture, and loader implementation.

Quantization can reduce memory but may introduce unsupported kernels, longer loading, quality changes, or serialization constraints. Keep a non-quantized/smaller-model fallback rather than assuming every advertised mode works on every GPU.

### LoRA

LoRA nodes load a compatible adapter and connect it to loader inputs that declare LoRA support. Base model, adapter architecture, target components, and scale must be compatible. A Hub search result alone does not establish that compatibility.

### Pipeline documentation

Modular nodes expose a `Doc` output where the upstream block provides documentation. Connect it to `Data Viewer` to inspect active blocks and accepted inputs. Treat live `/nodes` metadata as the executable contract when prose and the installed upstream revision disagree.

### Image comparison

Use the image comparison node to inspect outputs from different prompts, guiders, schedulers, resource modes, or model revisions. Record the model revision, seed, graph, and settings when the comparison is intended as reproducible evidence.

[Watch the image comparison demo (MP4)](https://github.com/user-attachments/assets/058fcce6-28db-4faf-9063-b48a0a1ea592)

## Troubleshooting

- Missing embeddings or dynamic fields usually mean the graph definition is stale or an upstream node did not produce its connected output. Update/recreate the graph after registry refresh.
- Missing scheduler/config errors require the compatible scheduler connection expected by the selected pipeline.
- CUDA OOM and unsupported quantized-kernel errors need a safer model/resource plan, not repeated blind retries.
- An unavailable optional package should be installed through the matching extra, followed by a backend restart.

See [docs/troubleshooting.md](../../docs/troubleshooting.md) for backend, accelerator, model-download, and bundle diagnostics.
