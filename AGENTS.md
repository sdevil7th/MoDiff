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
- Transformers and PEFT are required application dependencies installed by ordinary setup. Keep their heavyweight imports lazy and verify base-runtime readiness without optional installation or activation. Startup, registry discovery, template browsing, and Auto planning must never install packages. Additional optional runtimes still require an explicit install action and version verification.
- Ordinary deterministic media processing is allowed when narrowly scoped, documented, and covered by tests.
- Never enable arbitrary remote Python code, mutable model revisions, or custom model execution implicitly. Trust-sensitive behavior requires an explicit operator choice and an immutable revision.
- Prefer existing module, graph, configuration, error, and test patterns. Do not create a parallel workflow representation or model-loading path.

## Generic Workflow Stages

- Use content-aware initial sizes for new ordinary and custom nodes. Preserve explicit manual sizes and saved layouts. Check the editable body after the header and port tray are measured, including after model metadata arrives.
- Put an enabling switch before the fields it controls. Hide settings that the active operation ignores and preserve their values and connected inputs. Keep the primary model selector labelled and distinguish edit-in-place from conversion to independent nodes.
- Base filtered model choices and category showcases on actual available recipes and their approved media. Search long single- and multi-select lists, and resolve wire suggestions against the selected model's real socket contract.

- Keep search metadata with the backend visual stage catalog. Add modes, action names, familiar aliases and typed roles to an existing stage first. Verify name and action searches, click and wire insertion, model switching and saved graph export before adding another public stage.

- Reuse a task- or modality-generic node for the same user-visible job across models and workflows. Put model-specific execution and schema differences in the declared backend adapter.
- Use an ordinary node for a stage with one executable action. Expose the inputs, effective controls and outputs the user needs. Fold internal preparation or decoding into a stage only when it shares one owner and its intermediate result has no independent consumer.
- Reserve new Blocks for meaningful typed boundaries, such as storyboard shots, reusable compositions, loops or genuine nested control flow. Preserve existing saved Block interfaces and registered definition hashes.
- Keep compiler-only actions out of manual node discovery. Give public fields and sockets role-specific labels and show only controls consumed by the current operation. Test inactive saved values and existing connections when changing presentation.
- Label ports by role, including Source, Mask, Control, and First or Last Frame. Show one editor per effective setting and retain the saved fallback when a typed connection supplies the value. Choose a shared default only after it works in both the fresh workflow and the corresponding saved workflow.
- Resolve model variants that share a repository through the exact connected model and task declaration. Keep the canvas, restored workflow and saved output metadata consistent.
- Test the canvas and the ordered executable graph separately. Compare controls, typed bindings, model switching, and save/reopen. A presentation-only change preserves the executable graph and reuses the original run evidence.
- When unifying stages across model families or input types, test a real wire change and an older saved workflow in the packaged UI. Restore fixed model capabilities from the installed registry. Preserve the saved file fallback and each family’s settings across switches.
- Review the default blank-canvas picker separately from the execution registry. Start with generic stages and media processors. Keep specialist tools reachable by search, compatible wire or explicit implementation browsing. Compare the fresh template with its shipped Example and reopened execution graph. Check independent branches before folding preparation into a stage.

## Hugging Face Model Libraries

- Keep the reviewed Diffusers release pinned in the executable installation contract and record its artifact hashes in `uv.lock`. Keep immutable upstream catalog provenance separate from the installed distribution identity; update no-download compatibility tests and source-generated contracts when changing either.
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

## Custom extensions

- Read [Custom node development](docs/custom-nodes.md) before changing extension discovery, staging, enable, reload, or execution. Keep approvals outside source packages and bind them to inspected source and declared dependency versions.
- Executable custom Python uses explicit Add/Load/Reload and the existing executor. Bind source/dependency hashes internally; discovery, preview and graph import are not consent. Contract-only Blocks remain non-executable.
- A custom resource declaration is operator-reviewed code metadata, not catalog or hardware qualification. Do not execute custom suppliers during Auto inspection or assume their Python references are safe for early model eviction.

## Quality And Evidence

- Add a regression test that fails for the original defect and covers related instances of the same pattern.
- Keep public interfaces, errors, configuration, and non-obvious invariants documented. Examples must be runnable and must not require private files.
- Distinguish static inspection, unit/contract tests, HTTP smoke tests, and live model output. Never present one proof level as another.
- Report the exact commands and results you ran. A human maintainer remains responsible for understanding every changed line and reviewing generated content.

Run the focused tests while iterating, then use the complete backend gate documented in `CONTRIBUTING.md`. For changes that affect the client contract, also run the compatible MoDiff Client gate and its relevant browser tests.
