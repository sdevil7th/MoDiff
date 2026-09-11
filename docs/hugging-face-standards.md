# Hugging Face Engineering Alignment

MoDiff is not part of the Hugging Face organization, but its local model runtime may integrate official libraries maintained and published by Hugging Face. Diffusers and Modular Diffusers remain the original and primary integrations; additional libraries follow the same review, reproducibility, graph, and security requirements. This document records which upstream engineering expectations are project requirements and how contributors verify them.

## Upstream references

The project uses these maintained upstream sources as guidance:

- [Diffusers contribution guide](https://huggingface.co/docs/diffusers/main/en/conceptual/contribution), including its AI-assisted contribution rules
- [Diffusers design philosophy](https://huggingface.co/docs/diffusers/main/en/conceptual/philosophy)
- [Diffusers review rules](https://github.com/huggingface/diffusers/blob/main/.ai/review-rules.md)
- [Diffusers agent guide](https://github.com/huggingface/diffusers/blob/main/.ai/AGENTS.md)
- [Modular pipeline conventions](https://github.com/huggingface/diffusers/blob/main/.ai/modular.md)
- [Diffusers testing conventions](https://github.com/huggingface/diffusers/blob/main/.ai/testing.md)
- [Modular Diffusers overview](https://huggingface.co/docs/diffusers/en/modular_diffusers/overview)
- [Diffusers documentation source](https://github.com/huggingface/diffusers/tree/main/docs)
- [huggingface_hub contribution guide](https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md)
- [Transformers contribution guide](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md)
- [Hugging Face Dataset Cards](https://huggingface.co/docs/hub/en/datasets-cards)

Upstream repository layouts and release processes are not copied mechanically. MoDiff is an application with a Python server, a separate web client, hardware-specific environments, and persistent local workflows. The rules below adapt the common principles to those constraints.

## Required design principles

### Usability and explicit behavior

- Prefer a clear, composable path over a marginally faster but opaque implementation.
- Fail with an actionable error when a model, revision, artifact, input, package, or device is unsupported. Do not silently select a semantically different pipeline.
- Keep public graph, HTTP, WebSocket, and persistence contracts stable. A deliberate migration must update both repositories, tests, and documentation.
- Keep optional dependencies optional. Registry discovery and diagnostics must not import every model stack or require accelerator hardware.

### One graph boundary, reviewed Hugging Face runtimes

- A model-execution library is eligible when it is officially maintained and published by Hugging Face and its ownership and package provenance have been verified during integration. Eligibility is not automatic support: each library needs a declared use, reviewed dependency/version contract, backend adapter, and compatibility tests.
- Hugging Face Hub hosting does not make a library, model, or repository-supplied Python implementation first-party. Curated model and adapter references use immutable revisions where the Hub supports them, and remote code remains separately trust-gated.
- Every library executes locally through MoDiff's existing node graph, resource lifecycle, progress/cancellation, Auto/Expert, file, and output contracts. Do not add alternate graph executors, hosted inference providers, browser-side model runtimes, or a second workflow representation.
- Frontend nodes and task surfaces remain modality- or task-generic. The backend execution specification selects the reviewed library/model adapter and publishes its dynamic input, parameter, and output contract. The client must not infer Python classes or maintain a parallel model-specific parameter table.
- Transformers is an optional runtime and is not part of the default application installation. Registry discovery, template browsing/opening, and Auto planning may report that it is required but must not install it. Installation requires an explicit user action against a reviewed optional-runtime profile, followed by version and compatibility verification before execution.
- Accelerate, PEFT, quantization libraries, accelerator kernels, and ordinary deterministic image, audio, video, tensor, and file operations may support a model path when narrowly scoped and tested.
- Keep the reviewed Diffusers revision pinned in the executable installer contract. Apply an equally explicit compatible-version or immutable-source contract to every additional execution library.

Transition status: the legacy base dependency on Transformers and its required preflight check remain until roadmap segment P0.5 migrates existing Diffusers consumers to the staged optional-runtime contract. The policy above is the acceptance criterion for that migration, not a claim that the current installer already omits Transformers.

### Modular Diffusers

- Reuse upstream blocks, components, schedulers, loaders, adapters, and offload hooks instead of copying their behavior.
- Keep block inputs, outputs, and dependencies explicit and parser-friendly.
- Keep blocks composable and free of hidden cross-block state. Runtime caches belong to the existing memory/resource layer, not a second pipeline representation.
- Build executable pipelines through the upstream `init_pipeline` contract and cover the pinned upstream API with a no-download compatibility test.
- Require an explicit trust decision before executing repository-supplied Python. Never silently enable `trust_remote_code` or follow a moving revision.

### Focused and reviewable changes

- Diagnose the actual call path and compare similar implementations before editing.
- Keep a change focused on one outcome and remove unrelated generated files, proof scripts, and formatting churn from its diff.
- Add a regression test for the reported behavior and scan for other instances of the same pattern.
- Public functions and non-obvious contracts need concise documentation. Comments should explain constraints, not narrate the edit.

## Documentation and media

- Durable documentation lives under `docs/` and is linked from `docs/README.md`. Generated documentation output, dated work logs, host inventories, and implementation handoffs are not source documentation.
- Commands and examples must be runnable from a clean checkout and use repository-relative or placeholder paths.
- Documentation must distinguish registry/schema visibility, mocked behavior, local contract tests, HTTP smoke tests, and real model output.
- Large images, audio, and video do not belong in Git. Public template media is stored in the versioned public Hugging Face Dataset described by the client asset guide; Git stores only source descriptors, hashes, and small contracts.
- The Dataset card must describe contents, creation/provenance, intended use, limitations, licensing, privacy review, and versioning. A model license and the MoDiff Apache-2.0 code license do not automatically license generated media.

## AI-assisted contributions

AI tools may assist, but the human submitter owns the result. A change description must state the agreed scope, summarize the human self-review, and list exact validation commands and results. The submitter must understand every changed line, remove speculative or unrelated work, and must not treat an agent-generated claim as test evidence.

The repository-level instructions in [AGENTS.md](../AGENTS.md), contributor guide, pull-request template, and CI checks make these expectations visible to both people and tools.

## Evidence expected for a change

| Change | Minimum evidence |
| --- | --- |
| Pure documentation | Link/command review and repository policy checks |
| Backend logic | Focused regression test plus the complete backend test gate |
| HTTP, WebSocket, graph, or persistence contract | Backend contract tests and compatible client tests |
| Installer or dependency contract | Plan/dry-run tests for affected platforms, package validation, and a clean-profile smoke test where available |
| Modular Diffusers integration | Pinned upstream contract test, graph/schema round trip, and focused runtime tests |
| Template or public media | Gallery integrity/coverage checks, redacted provenance, Dataset manifest verification, and the applicable live proof level |
| Accelerator-specific behavior | Hardware-free contract coverage plus clearly identified live hardware evidence; unsupported hosts remain unclaimed |

No single evidence level substitutes for the others.
