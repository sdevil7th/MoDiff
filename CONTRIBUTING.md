# Contributing to MoDiff

MoDiff is an experimental local backend with a separately maintained frontend bundle. Contributions should preserve the local-only security boundary, existing graph/API compatibility, and reproducible dependency state.

Before starting, read [SECURITY.md](SECURITY.md) and the relevant guide in [docs/README.md](docs/README.md).

## Development setup

Use Python 3.12 and the committed lockfile:

```bash
uv sync --frozen
uv run python -m modiff.preflight --json --check-port 8088 --fail-on-error
```

Install only the extras required for the code path being changed. For example:

```bash
uv sync --frozen --extra quantization --extra spandrel
```

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

`pyproject.toml` and `uv.lock` are one change surface. After intentional metadata edits:

```bash
uv lock
uv lock --check
uv sync --frozen
uv pip check
```

Commit the updated lockfile. Keep platform markers and optional extras explicit, and update the README/configuration guidance when an install profile changes. If the manual pip fallback is affected, update the matching requirements file too.

## Validation

The baseline backend checks are:

```bash
uv lock --check
uv pip check
uv run python -m modiff.preflight --json --check-port 8088 --fail-on-error
uv run python -m unittest discover -s tests -v
```

On a host with Git Bash or a POSIX shell:

```bash
bash -n run.sh
```

Before reporting a live smoke test, confirm port `8088` is free or intentionally reuse the running process. A file-level inspection is not evidence that a model workflow completed; distinguish unit/contract tests, registry import checks, live backend HTTP checks, and real model-generation proof.

## Client-facing changes

The editable frontend source is not in this repository. It lives in the separate MoDiff-client checkout. Do not hand-edit minified files under `web/assets/`.

When a change affects the client/backend contract:

1. Update and validate the client source.
2. Run `npm ci` and `npm run check` in MoDiff-client.
3. Mirror the contents of its generated `dist/` directory into this repository's `web/` directory, deleting stale generated bundle files while preserving backend-owned `web/user/` custom fields.
4. Verify that `/`, `/assets/index.js`, and `/template-gallery/manifest.json` are served by a fresh backend.
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
