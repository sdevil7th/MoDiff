# MoDiff

MoDiff is a local client/server application for building and running node-based machine-learning workflows with a focus on [Hugging Face Diffusers](https://github.com/huggingface/diffusers). The backend discovers Python node modules, executes graphs, manages models and generated media, and serves a bundled web client from `web/`.

> [!CAUTION]
> MoDiff is early-stage software. It is not a production service, a multi-user platform, or a security sandbox. The server has no authentication and can execute model workflows, import custom Python modules, and access files inside its configured working directory. Keep it bound to `127.0.0.1`, install only code you trust, and read [SECURITY.md](SECURITY.md) before changing its network exposure.

## What is included

- A graph execution backend with queue, progress, interruption, cache, and structured runtime diagnostics.
- Diffusers-oriented image, audio, and video nodes, including Qwen Image, Wan VACE, Modular Diffusers, and reusable media/conditioning utilities.
- Hardware-aware resource planning for CUDA, Apple MPS, and CPU fallback.
- Hugging Face model discovery, download progress, cache diagnostics, and gated-model token setup.
- Local Studio output history, reusable blocks, workflow sharing, and a proof-backed template gallery.
- A prebuilt MoDiff client served by the backend at `http://127.0.0.1:8088`.

The registry currently spans these module groups:

`Audio`, `Color`, `DiffusersAudio`, `DiffusersImage`, `Experiments`, `Image`, `ImageFilters`, `ModelArtifact`, `ModularDiffusers`, `Primitive`, `QwenImage`, `Segmentation`, `Spandrel`, `Tensor`, `Text`, `Video`, `VideoColor`, `VideoConditioning`, and `WanVACE`.

Some nodes require optional packages, specific model repositories, substantial accelerator memory, or upstream experimental Diffusers APIs. Registry visibility does not by itself guarantee that every node is runnable on every machine.

## Repository layout

| Path           | Purpose                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| `modiff/`      | Canonical backend package.                                                                                           |
| `mellon/`      | Thin compatibility shims for older imports and automation. New code should use `modiff.*`.                           |
| `modules/`     | Built-in node implementations and registry metadata.                                                                 |
| `custom/`      | Locally installed custom Python modules; ignored except for repository placeholders.                                 |
| `data/graphs/` | Curated graph examples that are safe to version.                                                                     |
| `data/`        | Default local runtime data, model files, outputs, shares, and caches; most content is ignored.                       |
| `web/`         | Generated client bundle served by the backend. The editable frontend lives in the separate MoDiff-client repository. |
| `tests/`       | Backend contract and regression tests.                                                                               |

## Requirements

- Python 3.12.
- [`uv`](https://docs.astral.sh/uv/) for the recommended reproducible installation.
- Git for source checkout and optional custom-module installation.
- FFmpeg for video/audio workflows that use ImageIO or external codecs.
- A supported accelerator for practical use of larger models. CPU fallback exists, but many current workflows will be slow or impractical without CUDA.

Linux/Windows GPU resolution currently targets PyTorch CUDA 12.8 wheels. Install a compatible NVIDIA driver before expecting CUDA workflows to run. Apple Silicon uses normal PyPI PyTorch wheels and MPS when the installed PyTorch build and host support it.

## Quick start

From a fresh checkout:

```bash
uv sync --frozen
uv run python -m modiff.preflight --json --check-port 8088 --fail-on-error
uv run python main.py
```

Open <http://127.0.0.1:8088> after the server starts.

On Linux or macOS, `run.sh` selects the local virtual environment, then `uv`, then a system Python fallback:

```bash
chmod +x run.sh
./run.sh
```

On Windows, use PowerShell or Command Prompt:

```powershell
uv sync --frozen
uv run python -m modiff.preflight --json --check-port 8088 --fail-on-error
uv run python main.py
```

The defaults work without `config.ini`. To customize them, copy the example first:

```powershell
Copy-Item config.example.ini config.ini
```

```bash
cp config.example.ini config.ini
```

`config.ini` is intentionally ignored because it may contain a Hugging Face token and machine-local paths.

## Installation profiles

The lockfile is committed. Use `--frozen` for a reproducible checkout install, and name every optional extra needed by the workflows you intend to run.

| Extra                | Purpose                                                                                                                                  |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `apple-silicon`      | Explicit Apple Silicon/MPS installation profile using PyPI PyTorch packages.                                                             |
| `cuda`               | Adds CUDA-oriented optional acceleration packages such as `xformers`; base Linux/Windows resolution already selects CUDA PyTorch wheels. |
| `nunchaku`           | Installs the platform-specific Nunchaku wheel pinned by `pyproject.toml`.                                                                |
| `spandrel`           | Enables Spandrel upscaler nodes.                                                                                                         |
| `background-removal` | Enables `transparent-background` workflows.                                                                                              |
| `quantization`       | Adds DFloat11, GGUF, Quanto, TorchAO, kernels, and related quantization support where platform markers allow it.                         |
| `non-commercial`     | Adds `rembg[gpu]`; review its package/model licenses before redistribution or commercial use.                                            |

Examples:

```bash
# Apple Silicon
uv sync --frozen --extra apple-silicon

# Linux/Windows with optional CUDA acceleration and quantization
uv sync --frozen --extra cuda --extra quantization

# Add Nunchaku and Spandrel support
uv sync --frozen --extra nunchaku --extra spandrel
```

Avoid `--all-extras` unless you have reviewed the platform, build-tool, CUDA, and license requirements of every optional dependency.

### Manual pip fallback

`uv` is the maintained installation path. The requirements files are provided for manual environments, but they are not a cross-platform replacement for the lockfile.

On Linux, install the appropriate PyTorch build for the host first, then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install -U -r requirements.txt
python -m modiff.preflight --json --check-port 8088 --fail-on-error
python main.py
```

On Windows, create and activate the environment with PowerShell, then run the same install and startup commands:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel setuptools
python -m pip install -U -r requirements.txt
python -m modiff.preflight --json --check-port 8088 --fail-on-error
python main.py
```

Apple Silicon should use `requirements_macos.txt` instead of the CUDA-oriented base requirements:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install -U -r requirements_macos.txt
python -m modiff.preflight --json --check-port 8088 --fail-on-error
python main.py
```

The optional pip groups are `requirements_extras.txt` and `requirements_quant.txt`. Native extensions such as FlashAttention, SageAttention, Nunchaku, and some quantization backends may require their own supported PyTorch/CUDA combination and compiler toolchain.

## Configuration and local data

The complete commented example is [config.example.ini](config.example.ini). Important defaults:

- Server: `127.0.0.1:8088`, HTTP, CORS disabled.
- Working/data root: `data/` under the checkout.
- Model files: `data/models/`.
- Generated images/videos: `data/images/` and `data/videos/`.
- Studio history, blocks, and shares: `data/studio/`.
- Temporary files: `data/temp/`.
- Hugging Face cache: the normal Hugging Face cache unless `[huggingface] cache_dir` is set.

`[paths] work_dir` is the file-browser boundary. Relative data/model/output paths resolve beneath it. Give the process read/write access only to directories you intend MoDiff to use.

The Models UI can validate and save a Hugging Face read token. The token is written in plaintext to ignored `config.ini`; it is not encrypted or stored in an OS credential vault. Prefer a least-privilege read token, never commit the file, and rotate the token if it is exposed.

Generated outputs, prompts, workflow packages, and shares are also local plaintext files. Back up or remove them according to your own privacy and retention needs.

## Runtime and API diagnostics

Run preflight before investigating model-specific failures:

```bash
uv run python -m modiff.preflight --json --check-port 8088 --fail-on-error
```

The report checks Python, required imports, CUDA/MPS/CPU discovery, cache and data paths, and port state without importing the full node registry. The older `python -m mellon.preflight` command remains a compatibility entrypoint.

Useful local endpoints include:

- `GET /health` and `GET /runtime/status` for backend readiness.
- `GET /system_stats` for normalized hardware information.
- `GET /nodes` for the live node registry.
- `GET /queue` for current, queued, and recent task state.
- `GET /model_capabilities` and `GET /model_cache/diagnostics` for model/runtime planning.

See [docs/api-reference.md](docs/api-reference.md) for route groups and trust implications. These routes are designed for the bundled same-origin client and are not an authenticated public web API.

## Modular Diffusers and compatibility

The Modular Diffusers integration is documented in [modules/ModularDiffusers/README.md](modules/ModularDiffusers/README.md). It relies on experimental upstream APIs and intentionally retains external identifiers such as `MellonPipelineConfig`, `MellonParam`, and `diffusers.modular_pipelines.mellon_node_utils` where upstream compatibility requires them.

The physical `mellon` package is also retained as a thin import shim. New implementation belongs in `modiff`; compatibility policy and removal criteria are documented in [docs/modiff-backend-namespace.md](docs/modiff-backend-namespace.md).

## Updating

For a normal backend update:

```bash
git pull --ff-only
uv lock --check
uv sync --frozen
uv run python -m modiff.preflight --json --check-port 8088 --fail-on-error
uv run python -m unittest discover -s tests -v
```

If `pyproject.toml` is intentionally changed, regenerate and commit `uv.lock` with `uv lock`; do not bypass a stale-lock failure.

The bundled client does not update as part of `uv sync`. Client releases must be built in the separate MoDiff-client checkout and mirrored into this repository's `web/` directory.

## Updating the bundled client

Do not edit `web/assets/index.js` or `web/assets/index.css` by hand. They are generated artifacts.

1. Finish and validate changes in the MoDiff-client repository.
2. Run `npm ci` and `npm run check` there. The check command produces `dist/` after frontend tests and validation.
3. Mirror the **contents** of `dist/` into this backend's `web/` directory, deleting stale generated bundle files while preserving backend-owned `web/user/` custom fields. Do not create `web/dist/`.
4. Confirm that `web/index.html`, `web/assets/`, `web/favicon.ico`, and `web/template-gallery/` came from the same build, and that any existing `web/user/` directory survived the sync.
5. Start the backend and verify `/`, `/assets/index.js`, and `/template-gallery/manifest.json` before committing both repositories.

For adjacent checkouts, an exact mirror can be performed with a platform tool after confirming both paths:

```bash
# Run from MoDiff-client on Linux/macOS
rsync -a --delete --exclude '/user/' dist/ ../MoDiff/web/
```

```powershell
# Run from MoDiff-client on Windows. Preserve backend-owned web/user.
# Robocopy exit codes below 8 are success.
robocopy .\dist ..\MoDiff\web /MIR /XD user
if ($LASTEXITCODE -ge 8) { throw "Client bundle sync failed with exit code $LASTEXITCODE" }
```

Backend-only contributors who do not have the matching client checkout should leave `web/` unchanged.

## Troubleshooting and contributing

- [docs/README.md](docs/README.md) is the backend documentation index and recommended reading order.
- [docs/troubleshooting.md](docs/troubleshooting.md) covers startup, port, accelerator, model-download, FFmpeg, and stale-client problems.
- [CONTRIBUTING.md](CONTRIBUTING.md) describes backend validation, module conventions, lockfile updates, and client-bundle changes.
- [SECURITY.md](SECURITY.md) documents the supported local-only trust boundary and vulnerability reporting.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) defines the expected behavior in project spaces.

## License

MoDiff is distributed under the [Apache License 2.0](LICENSE). Individual models, datasets, custom modules, and optional dependencies may have separate licenses and usage restrictions; review them before downloading, redistributing, or using their outputs commercially.
