# Contributing to MoDiff

MoDiff is an experimental local backend with a separately maintained frontend bundle. Contributions should preserve the local-only security boundary, existing graph/API compatibility, and reproducible dependency state.

Before starting, read [SECURITY.md](SECURITY.md) and the relevant guide in [docs/README.md](docs/README.md).

## Development setup

Use Python 3.12 and create the same managed CPU profile used by baseline CI:

```bash
./install.sh --accelerator cpu --backend-only
uv pip install --python .venv/bin/python -r requirements/test.txt
./scripts/with-runtime-env.sh ./.venv/bin/python -m modiff.preflight --json --check-port 8088 --fail-on-error
```

On Windows PowerShell, use the corresponding managed commands:

```powershell
.\install.ps1 -Accelerator cpu -BackendOnly
uv pip install --python .venv/Scripts/python.exe -r requirements/test.txt
.\.venv\Scripts\python.exe -m modiff.preflight --json --check-port 8088 --fail-on-error
```

Choose the qualified accelerator profile relevant to a hardware-specific change and report that validation separately. Do not use `uv sync` or `uv run`: the project is intentionally `uv`-unmanaged because the installer, not the generic resolver, owns the executable Torch profile.

Do not commit `config.ini`, `.env` files, model caches, generated outputs, local logs, virtual environments, or test caches.

## Backend conventions

- Put backend framework implementation in `modiff/` and built-in node implementations in `modules/`.
- Preserve current HTTP/WebSocket and graph-storage contracts unless a deliberate migration includes client changes and contract tests.
- Keep hardware probes non-fatal and retain CPU fallback when CUDA or MPS discovery fails.
- Treat file access, custom-module installation, remote code, token handling, and mutating routes as security-sensitive changes.
- Avoid importing the full model registry from lightweight diagnostics such as preflight.

### Adding a node module

Built-in node packages live under `modules/<Name>/` and normally contain:

```text
modules/Example/
  __init__.py
  main.py
```

Use a thin initializer:

```python
from .main import *  # noqa: F401,F403
```

Define executable nodes in `main.py` as `NodeBase` subclasses. The registry inspects class metadata with the AST as well as live imports, so keep class-level `label`, `category`, `params`, and related metadata parser-friendly. Prefer module-level constants over environment-dependent expressions inside class metadata. Optional heavyweight dependencies should be imported as late as practical so an unavailable optional feature does not crash discovery of unrelated modules.

Add focused tests for registry visibility, constructor safety, field contracts, and any runtime behavior that can be validated without downloading a large model.

## Dependency changes

The selected file under `requirements/profiles/`, `pyproject.toml`, and `modiff/compatibility/accelerators.v1.json` jointly define the executable runtime contract. Keep direct wheel URLs hash-verified, keep remote source dependencies pinned to immutable revisions, and retain the exact reviewed Diffusers commit. After an intentional dependency or profile edit, rebuild the relevant managed profile:

```bash
./install.sh --accelerator cpu --backend-only --repair
uv pip check --python .venv/bin/python
```

Use the corresponding accelerator instead of `cpu` when the change affects CUDA, ROCm, or MPS. Update the compatibility manifest and public installation guidance only when the evidence supports the claim. MoDiff deliberately has no `uv.lock`; do not generate one or describe the top-level requirements files as a cross-platform lock.

MoDiff may integrate model libraries officially maintained and published by Hugging Face, but each library remains a separately reviewed execution dependency. Verify its upstream ownership, package provenance, license, supported version, loading behavior, and remote-code boundary. Hub hosting alone does not establish that a library or model is maintained by Hugging Face. Keep execution in the existing backend graph and expose task-generic contracts to the client.

Transformers-specific execution is opt-in. Do not add Transformers to the base application environment or install an optional runtime merely because a template is viewed, nodes are discovered, or Auto compatibility is planned. A requiring workflow must identify its versioned optional runtime, show an explicit install/consent action, perform installation outside graph execution, and re-run package and compatibility checks before the workflow can run.

## Validation

The baseline backend checks are:

```bash
uvx --from ruff==0.12.7 ruff check . --select E9,F
uv pip check --python .venv/bin/python
./scripts/with-runtime-env.sh ./.venv/bin/python -m modiff.preflight --json --check-port 8088 --fail-on-error
./scripts/with-runtime-env.sh ./.venv/bin/python -m pytest -q
```

The wrapper applies the installed accelerator profile's process environment before Python imports Torch. On Windows, run `.venv/Scripts/python.exe` directly in the equivalent commands.

On a host with Git Bash or a POSIX shell:

```bash
bash -n run.sh scripts/with-runtime-env.sh
```

Before reporting a live smoke test, confirm port `8088` is free or intentionally reuse the running process. A file-level inspection is not evidence that a model workflow completed; distinguish unit/contract tests, registry import checks, live backend HTTP checks, and real model-generation proof.

## Client-facing changes

The editable frontend source is not in this repository. It lives in the separate MoDiff-client checkout. Do not hand-edit minified files under `web/assets/`.

When a change affects the client/backend contract:

1. Update and validate the client source.
2. Run `npm ci` and `npm run check` in MoDiff-client.
3. Mirror the contents of its generated `dist/` directory into this repository's `web/` directory, deleting stale generated bundle files while preserving backend-owned `web/user/` custom fields.
4. Verify that `/` and `/assets/index.js` are served by a fresh backend. Verify `/template-gallery/manifest.json` only for an explicit offline/local Gallery build; a remote-asset build intentionally omits that directory and route.
5. Include the matching backend and client commit identifiers in the change description when the repositories are published separately.

See [README.md](README.md#updating-the-bundled-client) for exact-mirror examples.

## Documentation and change descriptions

Update public documentation when commands, extras, routes, storage behavior, trust boundaries, or compatibility guarantees change. Keep claims proportional to the evidence actually run.

A useful change description includes:

- The user-visible problem and intended behavior.
- Backend/client compatibility impact.
- Security or data-migration impact.
- Exact validation commands and results.
- Remaining hardware, model-access, or live-runtime gaps.
