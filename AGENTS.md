# MoDiff Backend Agent Instructions

Before node/Block/hierarchy, persistence/execution, qualification or cross-machine
integration work, read [Cluster engineering lessons and required procedure](docs/cluster-engineering-lessons.md)
completely. Treat its reproduction, preservation and proof-level checks as required,
not optional background. Keep the mirrored guide in both repositories aligned.

These rules apply to AI-assisted work in this repository. `CONTRIBUTING.md` is the complete contributor guide; read it together with `SECURITY.md` and the relevant document under `docs/` before editing.

## Scope And Runtime Boundary

- Keep each change focused on a diagnosed problem. Trace the existing call path and tests before editing, and remove incidental generated files from the diff.
- MoDiff may execute models through an official library maintained and published by Hugging Face. A Hub repository, organization name, or compatible API is not enough: verify the library's upstream ownership and package provenance, then declare and review the exact integration in MoDiff's executable dependency contract.
- All supported libraries run locally behind MoDiff's existing node graph, resource management, Auto/Expert, file, and security boundaries. Do not add an alternate graph executor, hosted inference provider, browser-side runtime, or library-owned workflow representation.
- Keep nodes and client contracts task- or modality-generic. Library- and model-specific loading, parameter aliases, and output normalization belong in small backend adapters selected from a declared execution specification, not new model-named nodes or frontend branches.
- Transformers is an optional runtime, not a default application dependency. Do not install it during startup, registry discovery, template browsing/opening, or Auto planning. A workflow that requires it must declare a reviewed optional runtime profile, present an explicit install/consent action, and verify the installed version before becoming runnable.
- Ordinary deterministic media processing is allowed when narrowly scoped, documented, and covered by tests.
- Never enable arbitrary remote Python code, mutable model revisions, or custom model execution implicitly. Trust-sensitive behavior requires an explicit operator choice and an immutable revision.
- Prefer existing module, graph, configuration, error, and test patterns. Do not create a parallel workflow representation or model-loading path.

## Hugging Face Model Libraries

- Keep the reviewed Diffusers revision pinned in the executable installation contract and update its compatibility test when changing it.
- Treat support for each additional official Hugging Face library as an explicit integration: document its purpose and provenance, constrain its compatible version, keep heavyweight imports lazy, and add no-download compatibility and boundary tests.
- Official-library eligibility is not blanket trust for Hub artifacts or repository code. Prefer `safetensors`; pin curated models and adapters immutably; and require a separate explicit operator decision for any reviewed remote-code path.
- Use Diffusers loaders, pipelines, components, schedulers, adapters, and offload hooks instead of reimplementing upstream behavior.
- Modular blocks should declare inputs, outputs, and dependencies clearly, avoid hidden cross-block state, and remain composable through `init_pipeline`.
- Keep model-specific differences explicit and small. Put reusable behavior in the existing shared Diffusers modules rather than copying it into another pipeline.
- For non-Diffusers libraries, put reusable behavior in the corresponding generic task module and preserve the same graph, resource, progress, cancellation, and output contracts.

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
