# Accelerator installation

MoDiff owns Python/Torch profile selection. The browser shows the same setup checklist but never installs drivers or mutates Python.

Run `./install.sh` on Linux/macOS or `.\install.ps1` on Windows. The guided installer explains every action before it runs, installs hash-verified app-local uv/Python 3.12, stages the backend, validates a real device tensor, and preserves the previous environment for rollback. When a sibling client is present and the build is not skipped, it downloads the verified Node 24 toolchain on demand before installing and building that client. Hybrid NVIDIA/AMD machines must explicitly choose a profile.

Useful modes:

- `--dry-run` or `--system-check`: explain the plan without changing anything.
- `--resume`: continue from the external `.modiff/install-state.json` journal.
- `--repair`: rebuild and validate the managed environment.
- `--non-interactive`: never invoke sudo/UAC; return structured required actions.
- `--backend-only`: skip sibling client installation and Node provisioning.
- `--guide`: print help for every stable setup error code.
- `--json`: return the same phases and structured steps for automation.

Profiles are `nvidia-cuda`, `amd-rocm-linux`, `amd-pytorch-windows`, `apple-mps`, and `cpu`. Strix Halo on Ubuntu 24.04.3 uses the AMD ROCm 7.2/PyTorch 2.9.1 profile. Ubuntu 26.04 is experimental and requires `--allow-experimental`; non-interactive Auto otherwise selects CPU. WSL and unqualified accelerators fall back to CPU.

Missing `video`/`render` membership, `/dev/kfd`, DRM render nodes, a successful `rocminfo` GPU agent, or the minimum kernel is blocking. Allowlisted administrator actions show their exact effect and require immediate confirmation. The installer saves state before offering a reboot and always prints the resume command.

The installer never changes firmware/BIOS or memory settings. It never executes commands received from the browser or backend API. Declined or unqualified GPU preparation offers the supported CPU command without deleting the GPU setup journal.
