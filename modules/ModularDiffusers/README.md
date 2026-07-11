# Modular Diffusers in MoDiff

MoDiff integrates the experimental [Diffusers Modular Pipelines](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/overview) APIs with its node graph. A small set of dynamic nodes can expose different model pipelines without creating a separate hardcoded node class for every model family.

> [!WARNING]
> Modular Diffusers APIs and compatible Hub repositories are still evolving. A visible node contract is not proof that every model/revision will load or fit on the current hardware. Custom blocks and `trust_remote_code` can execute repository-supplied Python; use only reviewed, revision-pinned sources and read [SECURITY.md](../../SECURITY.md).

## Concepts

- **Dynamic node contracts:** node fields adapt to the selected pipeline configuration.
- **Composable workflows:** model loading, prompt encoding, denoising, and decoding can remain separate or be combined into a custom block.
- **Shared components:** compatible nodes can reuse components from the package-level `ComponentsManager` instead of loading duplicate models.
- **Hub-backed blocks:** supported repositories can provide Modular Diffusers configuration/code used to construct a node interface.
- **Resource controls:** loaders expose supported quantization and offload modes, subject to package, model, and hardware compatibility.

Upstream APIs still use identifiers such as `MellonPipelineConfig`, `MellonParam`, and `mellon_node_utils`. Those names are external compatibility surfaces, not the active MoDiff product namespace. See [the namespace policy](../../docs/modiff-backend-namespace.md).

## Setup

Install and validate the backend from the repository root:

```bash
uv sync --frozen
uv run python -m modiff.preflight --json --check-port 8088 --fail-on-error
uv run python main.py
```

Open <http://127.0.0.1:8088>. Keep the server on loopback; it has no authentication or remote-code sandbox.

Optional quantization, Nunchaku, or other acceleration paths require the matching extras described in the [root README](../../README.md#installation-profiles).

## Start with a bundled graph

Open the workflow library in the left sidebar and expand `modular_diffusers`. The repository currently includes:

- `text_to_image` — a separated text-to-image pipeline.
- `image_to_image` — prompt plus reference-image conditioning.
- `multiple_image_edit` — multiple-image editing inputs.
- `quantization` — an example with an explicit quantization configuration.
- `dynamic_node` — a Hub-backed dynamic block example.

Drag a graph onto the canvas, inspect its selected model and required inputs, then update the graph before running. Models are not bundled with these JSON files; MoDiff may need to download them, and gated repositories may require accepted terms plus a Hugging Face read token.

[Watch the bundled workflow browser demo (MP4)](https://github.com/user-attachments/assets/a4d0604f-80ea-4470-80e6-53a73e584ca3)

## The five-node workflow

The bundled `text_to_image` graph illustrates five stages:

1. **Load Models** selects the pipeline type and exposes compatible text encoders, denoise model, VAE, scheduler, and optional image encoder components.
2. **Encode Prompt** converts prompt fields into the embeddings expected by the selected pipeline.
3. **Denoise** performs the iterative generation step using width, height, step, guidance, seed, and model-specific fields.
4. **Decode Latents** converts latent output into image output with the VAE.
5. **Preview Image** publishes the generated image to the client/cache surface.

The exact fields and defaults come from the live registry. For example, Flux, Qwen Image, Z-Image, and Wan pipelines do not share one universal guidance, prompt, or step contract. Refresh or recreate a graph when a model's dynamic definition changes.

[Watch a separated Modular Diffusers workflow demo (MP4)](https://github.com/user-attachments/assets/4bbf74ac-404e-46bb-ae51-a84e65c25235)

Type a prompt, confirm model readiness, and use **Run**. A queued task response only confirms submission; watch Queue and WebSocket progress for completion or structured failure details.

[Watch a workflow execution demo (MP4)](https://github.com/user-attachments/assets/e563eeb0-4f9e-4a27-8304-49fd15b87550)

## Reusing a loaded model

Compatible tasks can share components from one `Load Models` node. For example, an image-edit path can add image encoding/conditioning nodes while reusing the model components already loaded for text-to-image.

Component reuse depends on compatible pipeline contracts and current cache state. It reduces duplicate loading but does not guarantee that every model remains resident or that a new task avoids additional allocations.

[Watch a model-reuse demo (MP4)](https://github.com/user-attachments/assets/ddbc3e06-6254-4595-8209-4cfd98d3aabc)

## Dynamic Block

`Dynamic Block` combines a compatible Modular Diffusers block configuration into one graph node. Enter a supported repository ID, load its definition, inspect the generated fields, and connect any required shared components or media inputs.

For example, repositories such as `YiYiXu/FLUX.2-klein-4B-modular` have been used to demonstrate a compact prompt-to-image block. Repository availability and code can change; pin/review the intended revision before trusting it.

Dynamic blocks are not arbitrary no-code plugins. They must expose a structure understood by the current Diffusers/MoDiff integration, may require remote Python code, and can fail when upstream APIs or model files change.

## Combining workflows

Multiple Modular Diffusers paths can coexist on one canvas and share compatible components. A second block can consume the output of the first while reusing loaders, or separate denoise/decode paths can compare schedulers and resource modes.

Only nodes connected to the submitted graph path execute, but shared component state still consumes memory. Inspect Queue, loader diagnostics, and GPU-process information when a combined graph exceeds available resources.

## Custom Hub blocks

MoDiff can load compatible custom blocks from the Hugging Face Hub. This is a trust-sensitive feature:

1. Review the repository, owner, dependencies, license, and exact commit.
2. Prefer immutable revisions rather than a moving branch.
3. Enable `trust_remote_code` only when the repository requires it and you accept that its Python executes with backend-process permissions.
4. Test on a dedicated local environment without sensitive files in `work_dir`.

The example repository ID `diffusers/gemini-prompt-expander-mellon` intentionally retains an external legacy name. If available and compatible, it can generate a prompt-expansion block that feeds the prompt encoder. Its name should not be mechanically rewritten unless the Hub repository itself moves.

[Watch the custom prompt block demo (MP4)](https://github.com/user-attachments/assets/d68bc8c1-1b1c-478a-b94b-1e498c60a4fc)

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
