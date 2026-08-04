# MoDiff Backend Agent Instructions

These rules apply to AI-assisted work in this repository. `CONTRIBUTING.md` is the complete contributor guide; read it together with `SECURITY.md` and the relevant document under `docs/` before editing.

## Scope And Runtime Boundary

- Keep each change focused on a diagnosed problem. Trace the existing call path and tests before editing, and remove incidental generated files from the diff.
- Hugging Face Diffusers and Modular Diffusers are MoDiff's only supported model-execution layer. Do not add an alternate graph executor, hosted inference provider, independent Transformers application, or another model driver.
- Supporting libraries used by Diffusers and ordinary deterministic media processing are not alternate drivers. They must remain narrowly scoped, documented, and covered by tests.
- Never enable arbitrary remote Python code, mutable model revisions, or custom model execution implicitly. Trust-sensitive behavior requires an explicit operator choice and an immutable revision.
- Prefer existing module, graph, configuration, error, and test patterns. Do not create a parallel workflow representation or model-loading path.

## Diffusers And Modular Diffusers

- Keep the reviewed Diffusers revision pinned in the executable installation contract and update its compatibility test when changing it.
- Use Diffusers loaders, pipelines, components, schedulers, adapters, and offload hooks instead of reimplementing upstream behavior.
- Modular blocks should declare inputs, outputs, and dependencies clearly, avoid hidden cross-block state, and remain composable through `init_pipeline`.
- Keep model-specific differences explicit and small. Put reusable behavior in the existing shared Diffusers modules rather than copying it into another pipeline.
- Prefer `safetensors`; document and test any unavoidable unsafe deserialization or remote-code boundary.

## Security And Data

- Treat every HTTP, WebSocket, filesystem, archive, URL, model-repository, and workflow boundary as untrusted input.
- Mutations must not use `GET`. Resolve filesystem targets before access, reject traversal and symlink escapes, and keep request sizes finite.
- The supported server boundary is a trusted single user on `127.0.0.1`; do not weaken that default or imply that CORS is authentication.
- Never commit `config.ini`, tokens, local paths, generated outputs, model caches, qualification workspaces, virtual environments, logs, or template media.
- Public template media belongs in the configured public Hugging Face Dataset repository. Keep only its versioned source descriptor, hashes, and documentation in Git.

## Quality And Evidence

- Add a regression test that fails for the original defect and covers related instances of the same pattern.
- Keep public interfaces, errors, configuration, and non-obvious invariants documented. Examples must be runnable and must not require private files.
- Distinguish static inspection, unit/contract tests, HTTP smoke tests, and live model output. Never present one proof level as another.
- Report the exact commands and results you ran. A human maintainer remains responsible for understanding every changed line and reviewing generated content.

Run the focused tests while iterating, then use the complete backend gate documented in `CONTRIBUTING.md`. For changes that affect the client contract, also run the compatible MoDiff Client gate and its relevant browser tests.
