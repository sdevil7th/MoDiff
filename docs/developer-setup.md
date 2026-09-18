# Developer setup with uv and npm

Use these commands from the backend checkout on Linux or Windows PowerShell.
They invoke Python directly; no downloaded shell or PowerShell launcher is required.
Install Git and [uv 0.11.26](https://docs.astral.sh/uv/getting-started/installation/)
first. uv can provision Python 3.12. For the client, use Node 24.12.x and npm 11.6.2.

```text
uv run --no-project --no-sync --python 3.12 -m modiff.dev plan --accelerator cpu --backend-only --json
uv run --no-project --no-sync --python 3.12 -m modiff.dev setup --accelerator cpu --backend-only --non-interactive
uv run --no-project --no-sync --python 3.12 -m modiff.dev check --json --check-port 8088 --fail-on-error
uv run --no-project --no-sync --python 3.12 -m modiff.dev run
```

Open http://127.0.0.1:8088. The CPU profile is a small, no-model starting point for
API and UI development. It is not a claim that large diffusion models fit in CPU
memory. Setup downloads Python packages, including Torch; it does not download
inference weights. The backend-only path serves the checked frontend bundle.

`plan` only inspects the selected installer profile. `setup` delegates to the same
reviewed installer used by `install.sh` and `install.ps1`: exact Diffusers source,
platform Torch sources, package constraints, device smoke, staged promotion and
rollback are shared. Guided installation remains available.

An existing `.venv` is preserved. Use `check` to inspect it. To deliberately replace
its installed profile, repeat `setup` with `--repair` and the intended accelerator.
Do not repair your working GPU environment to CPU just to test the example; use a
separate checkout. `run` selects the installed interpreter and applies its ROCm
process environment before Torch imports. Ctrl+C stops the supervised runtime.

## Accelerator and optional runtime choices

Replace `cpu` with the appropriate installer selector: `auto`, `nvidia`, `amd`,
`amd-instinct`, `intel`, or `mps`. The authoritative support tiers, operating
systems and prerequisites are in the [accelerator guide](accelerator-installation.md).
An experimental profile still requires explicit `--allow-experimental` consent;
using uv does not qualify new hardware or bypass a missing system prerequisite.
`plan` reports the same required driver/system actions as guided setup.

MoDiff intentionally has no `uv.lock` and is marked `uv`-unmanaged. Ordinary
`uv run` and `uv sync` are not supported inside the accelerator environment.
[`--no-project`](https://docs.astral.sh/uv/reference/cli/#uv-run--no-project) disables
project discovery/resolution; `--no-sync` is redundant alongside it in uv 0.11.26
and uv prints a harmless warning. These explicit bootstrap commands do not ask
uv to resolve or replace the project's Torch profile. `uv pip` with an explicit
`--python` remains the supported contributor command for test dependencies.

Selected `requirements/profiles/*.txt`, `pyproject.toml`, and the accelerator
manifest jointly define the reviewed executable contract. They are not a complete
cross-platform transitive lock. The installer writes a profile receipt and checks
it at startup. Service export additionally records the observed package versions.
Optional runtimes retain their separate reviewed install, verification and
activation steps in Model Manager; opening a graph or exporting a service never
installs them. Keep credentials outside graph files and packages.

## Client development

From the sibling `MoDiff-client` checkout:

```text
npm ci
npm run dev
```

Use the URL printed by Vite with the backend running on its default loopback
address. `npm ci` consumes the committed lockfile. `npm run check` runs the client
quality gate and builds `dist/`; `npm run check:ui` runs browser regressions.
The backend's development command does not rebuild the client on every launch.
See [CONTRIBUTING](../CONTRIBUTING.md) for installing backend test requirements
and mirroring a validated production client bundle.

## Verification scope

The CPU matrix runs these setup/check commands and
`scripts/smoke_service_package.py` on Linux, Windows and macOS. The smoke redirects
all writable paths to temporary storage, binds an ephemeral loopback port, checks
health, and compares a saved model-free graph with Manual and Auto service calls.
A CI definition is not evidence of an executed Windows run. See the workbench
milestone tracker for the platforms actually validated in the current change.
See [service prototyping](service-prototyping.md) for the executable example.
