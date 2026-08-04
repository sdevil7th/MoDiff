## Summary

-

## Scope And Human Review

- Agreed scope / linked issue or discussion:
- Human self-review completed; every changed line and generated artifact is understood: yes / no
- AI assistance used (tool and role), or `none`:
- Follow-up work deliberately left out of this change:

## Checks

- [ ] `uvx --from ruff==0.12.7 ruff check . --select E9,F`
- [ ] `uv pip check --python .venv/bin/python` (use `.venv/Scripts/python.exe` on Windows)
- [ ] `./.venv/bin/python -m modiff.preflight --json --check-port 8088 --fail-on-error`
- [ ] `./.venv/bin/python -m pytest -q`
- [ ] `bash -n run.sh` when a POSIX shell is available
- [ ] Exact command results and any skipped checks are recorded below.

## Validation Results

- Commands and results:
- Checks skipped, with reason:
- Live accelerator/model evidence, or `not claimed`:

## Compatibility, Security, And Proof

- [ ] Current graph, HTTP, WebSocket, storage, and Python package contracts are preserved or deliberately migrated with tests.
- [ ] Client-facing changes include the matching MoDiff-client change and bundled-client update plan.
- [ ] File access, custom code, model loading, tokens, origins, and request-size implications were reviewed where relevant.
- [ ] Unit/contract, registry, live backend, accelerator, and real model-generation evidence are reported separately.
- [ ] Documentation was updated for changed commands, extras, routes, configuration, storage, or trust boundaries.
- [ ] No credentials, private media, personal paths, machine inventories, or unredacted provenance are included.
