## Summary

-

## Checks

- [ ] `uv lock --check`
- [ ] `uv pip check`
- [ ] `uv run python -m modiff.preflight --json --check-port 8088 --fail-on-error`
- [ ] `uv run python -m unittest discover -s tests -v`
- [ ] `bash -n run.sh` when a POSIX shell is available

## Compatibility, Security, And Proof

- [ ] Existing graph, HTTP, WebSocket, storage, and `mellon` import compatibility is preserved or the migration is described.
- [ ] Client-facing changes include the matching MoDiff-client change and bundled-client update plan.
- [ ] File access, custom code, model loading, tokens, origins, and request-size implications were reviewed where relevant.
- [ ] Unit/contract, registry, live backend, accelerator, and real model-generation evidence are reported separately.
- [ ] Documentation was updated for changed commands, extras, routes, configuration, storage, or trust boundaries.
- [ ] No credentials, private media, personal paths, machine inventories, or unredacted provenance are included.
