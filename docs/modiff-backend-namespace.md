# MoDiff Backend Namespace

The backend product and canonical Python namespace are **MoDiff** and `modiff`. New code, launchers, documentation, and user-facing logs should use those names.

The physical `mellon` package remains only as a compatibility layer. Its modules re-export canonical `modiff.*` objects so older scripts, custom nodes, and automation can migrate without maintaining two implementations.

## Canonical and compatibility entrypoints

| Purpose         | Canonical                    | Compatibility                |
| --------------- | ---------------------------- | ---------------------------- |
| Backend package | `modiff`                     | `mellon`                     |
| Node base       | `modiff.NodeBase`            | `mellon.NodeBase`            |
| Server          | `modiff.server`              | `mellon.server`              |
| Configuration   | `modiff.config`              | `mellon.config`              |
| Model store     | `modiff.modelstore`          | `mellon.modelstore`          |
| Client helpers  | `modiff.client`              | `mellon.client`              |
| Preflight       | `python -m modiff.preflight` | `python -m mellon.preflight` |

Compatibility wrappers should stay thin. Fixes belong in the canonical module and should be exercised through both import paths when identity or import order matters.

## Current compatibility rules

- Prefer `python -m modiff.preflight` for new diagnostics.
- Keep legacy `mellon.*` imports and `python -m mellon.preflight` working while the shim package is supported.
- Keep established HTTP and WebSocket contracts, including `/graph`, `/queue`, `/studio_outputs`, and `/workflows/share`, compatible with the separately versioned client.
- Normalize legacy saved graph paths to `data/graphs/modiff` when possible while retaining fallback reads for older `data/graphs/mellon` references.
- Preserve legacy error/metadata aliases at compatibility boundaries when older clients or automation may still read them.
- Treat upstream Diffusers names such as `MellonPipelineConfig`, `MellonParam`, and `diffusers.modular_pipelines.mellon_node_utils` as external API identifiers. They cannot be mechanically renamed inside MoDiff.
- Do not rename existing Hugging Face repository IDs or dataset paths containing `mellon` unless the external resource itself moves.

These compatibility strings are not evidence that the active backend implementation still lives under the old package. Conversely, removing old Git commit history does not authorize deleting compatibility surfaces or upstream copyright/license notices.

## Validation expectations

Namespace changes should prove all of the following in fresh Python processes:

- Canonical imports work before the full registry has been imported.
- Legacy wrappers resolve to the same classes, configuration object, and server singleton as canonical imports.
- Direct `modiff.NodeBase` and `mellon.NodeBase` imports do not leave a partially initialized module registry.
- Both preflight entrypoints produce the same canonical namespace metadata.
- Normal backend startup still discovers the full module registry.

Import-order regressions can be hidden by a test suite that imports `modules` first, so subprocess coverage is required for this boundary.

## Removal criteria

There is no scheduled removal release. The `mellon` shim package can be retired only after all of these are true:

- Backend entrypoints, local scripts, packaged launchers, public docs, and supported external integrations use `modiff`.
- Custom-node and pipeline authors have documented replacements for every MoDiff-owned legacy symbol.
- External Diffusers helper names have an upstream migration path or remain isolated behind an adapter.
- At least one released migration window includes deprecation messaging and compatibility tests.
- Saved graphs, output history, workflow shares, WebSocket state, and stable HTTP behavior remain usable after migration.

Until then, preserve the shim and document intentional legacy names rather than attempting a hard-zero text replacement.
