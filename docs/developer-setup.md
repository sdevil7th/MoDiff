# Developer setup with uv and npm

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
MoDiff does not require an exact uv version. uv provisions Python 3.12 from the
project configuration. Client development uses Node 24.12.x and npm 11.6.2.

## Native setup and launch

From the backend checkout, these commands work in Linux shells and Windows
PowerShell without invoking repository installer scripts:

```text
uv sync --extra cpu
uv run --extra cpu python -m modiff.preflight --json --check-port 8088 --fail-on-error
uv run --extra cpu python main.py
```

For NVIDIA, use `--extra cuda` in every command. For Intel, use `--extra xpu`.
On Apple Silicon, omit the extra: `uv sync` and `uv run python main.py` use the
PyPI wheel with MPS support. CPU is useful for API/UI development; it does not
make large models practical on machines with insufficient memory.

The committed `uv.lock` records the dependency resolution. Transformers and
PEFT are mandatory base libraries, so ordinary text-to-image and LoRA workflows
need no optional install or Activate step. Accelerator extras route Torch and
its companion wheels to their matching official index. Basic CUDA setup does
not require xFormers, bitsandbytes, or a Triton compiler on Windows.

Setup downloads Python packages, not model weights. Open <http://127.0.0.1:8088>
for the bundled app. Model downloads and gated-model access still use the normal
model-manager flow. Ctrl+C stops the supervised backend.

Keep the same accelerator extra on sync and run: running without it can reconcile
away packages belonging to the selected extra. `uv sync` reconciles an existing
`.venv`; test a different accelerator in a separate checkout rather than replacing
a working GPU stack. AMD vendor-wheel environments retain the specialized
[accelerator installation](accelerator-installation.md) path. Do not run a generic
CPU/CUDA sync over a reviewed ROCm environment.

## Explicit uv pip alternative

For an environment managed with direct pip commands, choose the Torch backend
explicitly as well as the project extra:

```text
uv venv --python 3.12
uv pip install --python .venv/bin/python --torch-backend=cpu -e ".[cpu]"
uv pip check --python .venv/bin/python
```

On Windows, replace `.venv/bin/python` with `.venv/Scripts/python.exe`. For NVIDIA,
use `--torch-backend=cu128 -e ".[cuda]"`. Run that environment's Python directly:
`python main.py` after activation, or its full executable path. Do not mix pip
changes with `uv run` auto-sync unless those changes are declared in the project.

## Client development

From the sibling `MoDiff-client` checkout:

```text
npm ci
npm run dev
```

Keep the backend running and open the URL printed by Vite. `npm ci` consumes the
committed npm lock. `npm run check` runs the quality gate and builds `dist/`;
`npm run check:ui` runs browser regressions. Backend launch does not rebuild the
client. See [CONTRIBUTING](../CONTRIBUTING.md) for backend test dependencies and
mirroring a validated production bundle while preserving `web/user/`.

## Upgrades and repair

Standalone uv installations support `uv self update`; package-manager installs
must use their package manager. See [Astral's upgrade instructions](https://docs.astral.sh/uv/getting-started/installation/#upgrading-uv).
MoDiff accepts the operator's uv; there is no application-specific version pin.

After pulling application changes, run `uv sync` with the same accelerator extra,
then preflight and restart the backend. An intentional package upgrade uses
`uv lock --upgrade-package transformers --upgrade-package peft`, then sync and
validate. `uv lock --upgrade` upgrades all compatible dependencies and needs the
full contributor gate. The exact Diffusers source is a compatibility exception:
current Modular block contracts bind that revision; update it together with those
contracts and tests. Compiled accelerator wheels also need matching ABI versions.

A missing base library is repaired by rerunning the same sync command, without
optional activation. Additional optional packages retain explicit install and
activation, with progress through verification and worker replacement. Failed
or stale overlays must be repaired through their own runtime action; deleting
models, workflows, or output history is unnecessary.

## Windows allocator and compilation

MoDiff uses PyTorch's default allocator on Windows and does not inject
`expandable_segments:True`. Explicit `PYTORCH_ALLOC_CONF` or the legacy
`PYTORCH_CUDA_ALLOC_CONF` setting is preserved; remove unsupported allocator
options if your PyTorch build warns. Linux launch paths share the same pre-import
allocator policy.

Ordinary image workflows use eager execution and native SDPA without Triton.
Compilation is optional and requires an executed kernel probe. A workflow that
requires compiled FlexAttention checks its compiler before resolving or loading
model weights. Windows compilation uses a compatible
[triton-windows toolchain](https://github.com/triton-lang/triton-windows), whose
PyTorch compatibility must be checked separately; package import alone does not
prove that kernels work. A failed probe leaves ordinary eager workflows usable.

## Verification scope

CPU CI runs native setup, preflight, tests, and the model-free service smoke on
Linux, Windows, and macOS. An unexecuted CI definition is not proof of a Windows
GPU run. Real model qualification must record the OS, accelerator, driver,
packages, model, and repeated successful outputs. See
[service prototyping](service-prototyping.md) for the model-free example.
