<!-- Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff. -->

# MoDiff

MoDiff is a local client/server application for building and running node-based machine-learning workflows with a focus on [Hugging Face Diffusers](https://github.com/huggingface/diffusers). The backend discovers Python node modules, executes graphs, manages models and generated media, and serves a bundled web client from `web/`.

> [!CAUTION]
> MoDiff is early-stage software. It is not a production service, a multi-user platform, or a security sandbox. The server has no authentication and can execute model workflows, import custom Python modules, and access files inside its configured working directory. Keep it bound to `127.0.0.1`, install only code you trust, and read [SECURITY.md](SECURITY.md) before changing its network exposure.

## Before you install

MoDiff is distributed as paired backend and client source checkouts. For normal
use you need:

- A supported 64-bit Windows, Linux, or Apple Silicon macOS host.
- Git and an internet connection for the first installation and model downloads.
- FFmpeg on `PATH` for workflows that read or export video or audio.
- Enough free disk space for the application, model cache, temporary files, and
  outputs. Individual model repositories can require many gigabytes.
- A supported accelerator for practical use of larger models. CPU fallback is
  available, but many image workflows will be slow and most large video/audio
  workflows will be impractical.

The guided installer provisions its pinned `uv`, Python 3.12, and Node.js
toolchains when the selected platform supports bootstrapping them. The managed
NVIDIA profile uses the reviewed PyTorch CUDA 12.8 wheels. Qualified Linux AMD
hosts use the pinned ROCm profile. AMD's selected Windows stack is represented
as a conditional target but remains blocked until MoDiff pins and validates its
complete SDK wheel set. Intel Arc and supported Intel integrated graphics use
the preview XPU profile on x86-64 Linux/Windows. Apple Silicon uses reviewed
PyPI PyTorch wheels and MPS. Review the [runtime support matrix](docs/runtime-support-matrix.md)
and [accelerator installation guide](docs/accelerator-installation.md) before
choosing a non-default profile.

## Install and run

Clone both repositories into the same parent directory. The directory names and
sibling layout below let the backend installer find, build, and bundle the
client automatically.

Linux or macOS:

```bash
git clone https://github.com/sdevil7th/MoDiff.git MoDiff
git clone https://github.com/sdevil7th/MoDiff-client.git MoDiff-client
cd MoDiff
./install.sh --accelerator auto --system-check
./install.sh --accelerator auto
./run.sh
```

Windows PowerShell:

```powershell
git clone https://github.com/sdevil7th/MoDiff.git MoDiff
git clone https://github.com/sdevil7th/MoDiff-client.git MoDiff-client
cd MoDiff
.\install.ps1 -Accelerator auto -SystemCheck
.\install.ps1 -Accelerator auto
.\run.ps1
```

The system-check step reports the proposed accelerator profile and blockers
without installing packages. Review it, then continue with the normal installer
shown on the next line.

Open <http://127.0.0.1:8088>. The first installation downloads Python, Node.js,
packages, and the verified Template Gallery assets, so it can take time and use
substantial disk space. Normal Gallery browsing then uses the local bundled
files instead of downloading media during use.

`run.sh` and `run.ps1` start the backend in the foreground. The backend also
serves the installed frontend bundle, so this starts the complete application;
no separate frontend process is required. Press `Ctrl+C` in that terminal to
stop it.

The client's `run-dev.sh` and `run-dev.ps1` launch a separate Vite frontend for
hot reload and are intended only for frontend development. Their matching
shutdown commands are `stop-dev.sh` and `stop-dev.ps1`.

### Verify the installation

With `run.sh` or `run.ps1` still running, verify the backend from another
terminal:

```bash
curl --fail http://127.0.0.1:8088/health
curl --fail http://127.0.0.1:8088/runtime/status
```

```powershell
Invoke-RestMethod http://127.0.0.1:8088/health
Invoke-RestMethod http://127.0.0.1:8088/runtime/status
```

Then open <http://127.0.0.1:8088> and check these items:

1. The top-bar connection indicator reports that the backend is connected.
2. **Setup** shows the selected runtime profile and no unresolved environment
   repair blocker.
3. **Nodes** loads the live registry and **Models** can inspect the configured
   model locations.

A successful health check proves that the service is responding. It does not
prove that a particular model is installed, licensed, compatible with the
machine, or fast enough for practical use.

### Run your first workflow

1. Open **Setup** and resolve any runtime or model-cache blocker first.
2. If a selected Hugging Face repository is gated, accept its terms on Hugging
   Face and save a least-privilege read token through **Models**. The token is
   stored as plaintext in ignored `config.ini`.
3. Choose **Text to image** or a small image recipe from **Browse recipes**.
   Prefer a recipe that Auto marks ready for the detected device; do not use a
   large video model as the first installation test.
4. Keep the resource-mode **Auto** switch enabled, enter the required prompt and
   inputs, and review the readiness result.
5. Use the attached **Install** or **Repair** action when the selected artifact
   is missing or incomplete. Wait for its terminal status rather than assuming
   an unchanged download percentage is a failure.
6. Choose one-shot **Run**. Follow model preparation, node, and step progress in
   the session shelf or **Queue**.
7. Open **Gallery** after completion to inspect, download, restore, or reuse the
   output. Export important workflows instead of relying only on browser state.

The client [Studio user guide](https://github.com/sdevil7th/MoDiff-client/blob/main/docs/studio-user-flow.md)
explains the complete guided workflow, Auto/Expert controls, Queue, Gallery,
and recovery behavior.

### What Auto does—and does not do

Auto chooses an eligible runnable resource recipe using the current backend,
artifact, device, memory, and available qualification evidence. It may select a
dtype, pre-quantized artifact, offload strategy, attention backend, or other
known-safe runtime option. Unqualified recipes remain labeled as such. Auto does not promise the fastest recipe, benchmark
the model before submission, or make every catalog entry runnable.

In particular, a shared-memory or integrated GPU reporting a large addressable
memory pool is not equivalent to a discrete GPU with the same amount of local
VRAM. A workflow can be technically valid while model placement and every
denoising step remain very slow. The first run can also spend substantial time
downloading, validating, loading, and placing weights before generation starts.

The same prompt and generation parameters can sometimes run faster with a
different qualified runtime recipe—for example, a compatible pre-quantized
artifact, supported attention kernel, compilation/cache path, or improved
device placement. These choices normally require a new pipeline load and a new
run; they cannot safely accelerate work already in progress. Do not enable an
unqualified runtime quantizer or optional kernel merely because it is visible.

## What is included

- A graph execution backend with queue, progress, interruption, cache, and structured runtime diagnostics.
- Diffusers-oriented image, audio, and video nodes, including Qwen Image, Wan VACE, Modular Diffusers, and reusable media/conditioning utilities.
- Hardware-aware resource planning for NVIDIA CUDA, AMD ROCm, Apple MPS, and CPU fallback.
- Hugging Face model discovery, download progress, cache diagnostics, and gated-model token setup.
- Local Studio output history, reusable blocks, workflow sharing, and a proof-backed template gallery.
- A prebuilt MoDiff client served by the backend at `http://127.0.0.1:8088`.

The registry currently spans these module groups:

`Audio`, `Color`, `DiffusersAdapters`, `DiffusersAudio`, `DiffusersImage`, `DiffusersRuntime`, `DiffusersVideo`, `Image`, `ImageFilters`, `MediaSource`, `ModelArtifact`, `ModularDiffusers`, `Primitive`, `Spandrel`, `Tensor`, `Text`, `Video`, `VideoColor`, `VideoConditioning`, and `WorkflowControl`.

Some nodes require optional packages, specific model repositories, substantial accelerator memory, or upstream experimental Diffusers APIs. Registry visibility does not by itself guarantee that every node is runnable on every machine.

The deterministic edge, sketch, and optical-flow actions in `VideoConditioning` use the optional `gallery-media` OpenCV extra. Resize, mask alignment, frame selection, and authored shot-list controls remain available in the core install and do not load a separate model runtime.

## Repository layout

| Path           | Purpose                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| `modiff/`      | Canonical backend package.                                                                                           |
| `modules/`     | Built-in node implementations and registry metadata.                                                                 |
| `custom/`      | Locally installed custom Python modules; ignored except for repository placeholders.                                 |
| `data/graphs/` | Curated graph examples that are safe to version.                                                                     |
| `data/`        | Default local runtime data, model files, outputs, shares, and caches; most content is ignored.                       |
| `web/`         | Generated client bundle served by the backend. The editable frontend lives in the separate MoDiff-client repository. |
| `tests/`       | Backend contract and regression tests.                                                                               |

## Configuration

MoDiff is distributed as paired backend and client source checkouts, not as a
standalone Python wheel. The defaults work without `config.ini`. To customize
them, copy the example first:

```powershell
Copy-Item config.example.ini config.ini
```

```bash
cp config.example.ini config.ini
```

`config.ini` is intentionally ignored because it may contain a Hugging Face token and machine-local paths.

## Managed installation profiles

MoDiff's installer owns the executable Python/Torch environment. The project is intentionally marked `uv`-unmanaged, so `uv sync` and `uv run` are not supported setup or launch commands. The installer stages a fresh environment, checks its package policy and a real device tensor, then atomically promotes it to `.venv/` while retaining the previous environment for rollback. When the sibling client is installed, setup also downloads and SHA-256 verifies the pinned rights-approved Template Gallery snapshot and bundles it under `web/template-gallery` so normal use does not wait on Hub media requests. Four permission-dependent preview files are currently unavailable; their templates remain usable and do not request those files.

| Installer choice | Managed profile | Current scope |
| --- | --- | --- |
| `auto` | Host-dependent | Selects a qualified profile or a safe CPU fallback. |
| `nvidia` | `nvidia-cuda` | Linux/Windows NVIDIA with the reviewed CUDA 12.8 PyTorch profile. |
| `amd` | OS-dependent AMD profile | Qualified Linux AMD/ROCm hosts. Windows is a conditional official platform, but MoDiff blocks installation until the complete SDK wheel set and physical proof are pinned. |
| `intel` | `intel-xpu` | Preview PyTorch XPU profile for supported Intel Arc and integrated graphics on x86-64 Linux/Windows. |
| `mps` | `apple-mps` | Apple Silicon using the reviewed MPS-capable PyTorch profile. |
| `cpu` | `cpu` | Portable CPU environment for development and fallback. |

If an installation was interrupted, resume its external journal instead of
starting unrelated setup work:

```bash
./install.sh --accelerator auto --resume
```

```powershell
.\install.ps1 -Accelerator auto -Resume
```

Rebuild and validate the selected managed profile when preflight reports
runtime-contract drift or a repair requirement:

```bash
./install.sh --accelerator auto --repair
```

```powershell
.\install.ps1 -Accelerator auto -Repair
```

Backend contributors who intentionally do not want to build the sibling client
can install a CPU development environment with:

```bash
./install.sh --accelerator cpu --backend-only
```

```powershell
.\install.ps1 -Accelerator cpu -BackendOnly
```

See [the accelerator installation guide](docs/accelerator-installation.md) for
dry-run, non-interactive, resume, repair, and experimental-host behavior. The
selected file under `requirements/profiles/`, `pyproject.toml`, and the
accelerator compatibility manifest jointly define the reviewed runtime
contract.

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
./scripts/with-runtime-env.sh ./.venv/bin/python -m modiff.preflight --json --check-port 8088 --fail-on-error
```

On Windows, use `.\.venv\Scripts\python.exe` in place of `./.venv/bin/python`.

The report checks Python, required imports, CUDA/ROCm/MPS/CPU discovery, cache and data paths, and port state without importing the full node registry.

Useful local endpoints include:

- `GET /health` and `GET /runtime/status` for backend readiness.
- `GET /system_stats` for normalized hardware information.
- `GET /nodes` for the live node registry.
- `GET /queue` for current, queued, and recent task state.
- `GET /model_capabilities` and `GET /model_cache/diagnostics` for model/runtime planning.

See [docs/api-reference.md](docs/api-reference.md) for route groups and trust implications. These routes are designed for the bundled same-origin client and are not an authenticated public web API.

## Modular Diffusers

The Modular Diffusers integration is documented in [modules/ModularDiffusers/README.md](modules/ModularDiffusers/README.md). MoDiff owns the pipeline configuration schema used by its dynamic node contracts while relying on upstream Diffusers for model and pipeline execution.

## Updating and recovery

Stop the foreground application with `Ctrl+C` and update both sibling
repositories so the frontend and backend contracts remain aligned.

Linux or macOS, from their parent directory:

```bash
git -C MoDiff-client pull --ff-only
git -C MoDiff pull --ff-only
cd MoDiff
./install.sh --accelerator auto --repair
./run.sh
```

Windows PowerShell, from their parent directory:

```powershell
git -C .\MoDiff-client pull --ff-only
git -C .\MoDiff pull --ff-only
Set-Location .\MoDiff
.\install.ps1 -Accelerator auto -Repair
.\run.ps1
```

Use the accelerator you deliberately selected instead of `auto` when retaining
an explicit profile. The installer detects changes across profile
requirements, `pyproject.toml`, and the accelerator manifest, then rebuilds the
sibling client bundle during a normal paired installation. There is no
repository `uv.lock` to regenerate. A `--backend-only`/`-BackendOnly` repair
does not update the installed client.

For failures, begin with [troubleshooting](docs/troubleshooting.md) rather than
deleting caches or data. The safe-cleanup section identifies disposable files
and explains why `data/`, `config.ini`, and shared Hugging Face caches require
individual review and backups.

## Updating the bundled client

Do not edit `web/assets/index.js` or `web/assets/index.css` by hand. They are generated artifacts.

1. Finish and validate changes in the MoDiff-client repository.
2. Run `npm ci` and `npm run check` there. The check command produces `dist/` after frontend tests and validation.
3. Mirror the **contents** of `dist/` into this backend's `web/` directory, deleting stale generated bundle files while preserving backend-owned `web/user/` custom fields. Do not create `web/dist/`.
4. Confirm that `web/index.html`, `web/assets/`, and `web/favicon.ico` came from the same build, and that any existing `web/user/` directory survived the sync. A source-release build intentionally has no `web/template-gallery/` directory; the normal installer adds the verified local payload.
5. Start the backend and verify `/`, `/assets/index.js`, and `/template-gallery/manifest.json`. Lightweight source-release builds without an installer pass retain immutable Hub URL resolution.

The normal installer materializes `web/template-gallery/` for that
installation. Treat it as downloaded runtime data: do not add it to Git or a
normal remote-asset release package.

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
- [CONTRIBUTING.md](CONTRIBUTING.md) describes backend validation, module conventions, managed-runtime updates, and client-bundle changes.
- [SECURITY.md](SECURITY.md) documents the supported local-only trust boundary and vulnerability reporting.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) defines the expected behavior in project spaces.

## License

MoDiff source code is distributed under the [Apache License 2.0](LICENSE) and retains the copyright notice from the
upstream Mellon project from which it was derived. Model weights, adapters, and Gallery media are not relicensed by
MoDiff and remain subject to their respective upstream terms. Datasets, custom modules, and optional dependencies may
also have separate licenses and usage restrictions; review them before downloading, redistributing, or using their
outputs commercially. Attribution for the adapted Diffusers helper and licenses for font software
redistributed with the bundled client are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
