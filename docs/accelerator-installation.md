# Accelerator installation

MoDiff owns Python/Torch profile selection. The browser shows the same setup checklist but never installs drivers or mutates Python.

Run `./install.sh` on Linux/macOS or `.\install.ps1` on Windows. The guided installer explains every action before it runs, installs hash-verified app-local uv/Python 3.12, stages the backend, validates a real device tensor, and preserves the previous environment for rollback. When a sibling client is present and the build is not skipped, it downloads the verified Node 24 toolchain, downloads and SHA-256 verifies the complete pinned Template Gallery, and bundles those assets into the local client build. Hybrid NVIDIA/AMD machines must explicitly choose a profile.

Useful modes:

- `--dry-run` or `--system-check`: explain the plan without changing anything.
- `--resume`: continue from the external `.modiff/install-state.json` journal.
- `--repair`: rebuild and validate the managed environment.
- `--non-interactive`: never invoke sudo/UAC; return structured required actions.
- `--backend-only`: skip sibling client installation and Node provisioning.
- `--guide`: print help for every stable setup error code.
- `--json`: return the same phases and structured steps for automation.

On Windows, both PowerShell launchers preserve native progress written to
stderr when output is redirected (for example,
`.\install.ps1 -Accelerator auto *> install.log`). They use the native exit
code to detect failure; a successful progress message must not abort setup.
`run.ps1` still refuses to start the worker when profile validation fails.

Executable profiles are `nvidia-cuda`, `amd-rocm-linux`, `amd-instinct-rocm-linux` (preview), `intel-xpu`, `apple-mps`, and `cpu`. Choose Intel explicitly with `--accelerator intel`; Auto selects it when a supported Intel GPU is detected and no higher-priority NVIDIA/AMD profile applies. The XPU profile is preview-only and requires a successful `xpu:0` tensor before launch. The manifest retains `amd-pytorch-windows` as a conditional target, but installation remains blocked until MoDiff pins the complete official Windows SDK wheel set instead of guessing dependencies. Strix Halo on Ubuntu 24.04.3 uses the AMD ROCm 7.2/PyTorch 2.9.1 profile. The separate Instinct preview below targets MI300X/gfx942, not Ryzen. Ubuntu 26.04 is experimental and requires `--allow-experimental`; non-interactive Auto otherwise selects CPU. WSL and unqualified accelerators fall back to CPU.

On macOS, the `cpu` and `apple-mps` profiles use the same MPS-capable PyTorch wheel. The managed profile remains authoritative: selecting `cpu` validates and records a CPU tensor even though the wheel contains MPS support, while selecting `mps` requires an available MPS device and validates on `mps:0`.

Integrated AMD and Intel devices use shared system memory. Hardware discovery records accessible, dedicated, shared, and planning memory separately; Auto budgets from local/planning capacity instead of treating a large GTT or unified-memory aperture as equivalent discrete VRAM. MPS and XPU use direct device residency because CUDA-oriented Diffusers CPU-offload hooks are not portable to those backends.

Missing `video`/`render` membership, `/dev/kfd`, DRM render nodes, a successful `rocminfo` GPU agent, or the minimum kernel is blocking. Allowlisted administrator actions show their exact effect and require immediate confirmation. The installer saves state before offering a reboot and always prints the resume command.

The installer never changes firmware/BIOS or memory settings. It never executes commands received from the browser or backend API. Declined or unqualified GPU preparation offers the supported CPU command without deleting the GPU setup journal.

## Instinct MI300X cloud preview

The separate `amd-instinct-rocm-linux` profile targets `gfx942` on a
provider-prepared Ubuntu 24.04 host. Start with AMD's PyTorch 2.10.0 / ROCm
7.14 Quick Start image and use Python 3.12. This is a preview runtime contract,
not a claim of completed MI300X model qualification.

```bash
./install.sh --accelerator amd-instinct --backend-only --system-check --json
./install.sh --accelerator amd-instinct --backend-only --non-interactive
```

The first command is read-only. The second stages a fresh managed environment;
it does not reuse or overwrite the provider's global Python environment. The
profile pins Torch, torchvision, torchaudio, Triton and the gfx942/ROCm Python
SDK packages from AMD's index. Its adjacent `.uv.toml` is part of the runtime
contract: it permits fallback from AMD's public index on a missing-package
HTTP 403 while retaining default first-index selection and TLS verification.
This does not relax Hugging Face access or license checks. It does not invoke the Ryzen driver's installer
or inject the local Ryzen `/opt/rocm` library path. A real `cuda:0` tensor,
`gfx942` architecture, pinned Torch/HIP versions and package checks must pass
before the new environment is promoted. `--accelerator amd` also selects this
profile when `rocminfo` reports gfx942. Other Instinct architectures and mixed
GPU architectures are not automatically admitted.

Keep the application bound to `127.0.0.1` and access it over an SSH tunnel.
Use separate data and Hugging Face caches for cloud runs. Pull model revisions
through the app's Hub flow; never copy local credentials/workflow history in a
source snapshot. Provisioning, downloads and idle time are billable. Back up
outputs before destroying the instance; powering it off does not stop billing.
See [the cloud qualification checklist](amd-cloud-qualification.md).

The selected file under `requirements/profiles/`, `pyproject.toml`, and the
accelerator compatibility manifest jointly form the saved runtime contract.
Preflight recomputes that contract on every launch. If any input changed, is
missing, or the saved profile predates the contract digest, the runtime status
is `repair-required` and startup remains blocked even when the currently
installed packages still import. Run the reported managed `--repair` command;
do not bypass the check with a generic dependency sync.

## Maintaining direct-wheel profiles

`scripts/lock_accelerator_wheels.py` recalculates SHA-256 hashes only for a
profile whose requirements file consists of direct HTTPS wheel URLs. It
rejects index-based, ordinary pinned, placeholder, and unknown profiles before
writing. Run it from the repository root, review every upstream URL first, and
then synchronize the reviewed digests in
`modiff/compatibility/accelerators.v1.json`:

```bash
python scripts/lock_accelerator_wheels.py --profile amd-rocm-linux
```

Changing a wheel URL or digest is a runtime-contract change. Rebuild that
managed profile and rerun the accelerator manifest, installer, preflight, and
physical-device qualification checks before release.

On Linux, invoke developer preflight and test commands through
`./scripts/with-runtime-env.sh`. The wrapper reads the installed profile and
applies the same native runtime-library environment used by `run.sh`; this is
required for ROCm wheels whose shared libraries live below `/opt/rocm`.
