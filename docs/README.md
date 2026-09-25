# MoDiff Backend Documentation

This directory contains the durable technical guides for the MoDiff backend. Start with the root [README](../README.md) for installation and first launch, then use the guides below for the area you are working on.

## Read By Goal

| Goal                                                                     | Guide                                                                                       |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| Install, verify, run a first workflow, configure, or update MoDiff       | [Project README](../README.md) and [`config.example.ini`](../config.example.ini)            |
| Understand HTTP and WebSocket surfaces                                   | [API reference](api-reference.md)                                                           |
| Diagnose startup, ports, slow/stalled runs, devices, downloads, or media | [Troubleshooting](troubleshooting.md)                                                       |
| Compare the qualified accelerator profiles and their proof levels        | [Runtime support matrix](runtime-support-matrix.md)                                         |
| Check quantized model, dependency, download, and qualification support   | [Quantization support matrix](quantization-support.md)                                      |
| Review optional attention, quantization, and compilation capabilities    | [Optional runtime optimizations](optional-runtime-optimizations.md)                         |
| Build Modular Diffusers graphs and understand experimental compatibility | [Modular Diffusers guide](../modules/ModularDiffusers/README.md)                            |
| Use ordinary image actions and their optional typed inputs               | [Ordinary Diffusers image nodes](../modules/DiffusersImage/README.md)                       |
| Run the qualified image demo                                             | [Image demo checkpoint](image-demo.md)                                                      |
| Rehearse editable image stages and portable custom art direction          | [Image modularity demo](modularity-demo.md)                                                 |
| Build a visually meaningful fashion-editing demonstration                 | [SoHo fashion editorial demo](fashion-editorial-demo.md)                                    |
| Transfer the fashion demo to a Windows NVIDIA machine                     | [Windows fashion demo setup](windows-fashion-demo.md)                                       |
| Generate/edit with Qwen 2.1 and understand attention-context reuse       | [Qwen-Image 2.1](qwen-image-21.md)                                                          |
| Estimate depth with generic Transformers nodes                           | [Transformers depth workflows](../modules/HuggingFaceTransformers/README.md)                |
| Build reusable attention-mask and LoRA-scale inputs                      | [Attention Arguments](../modules/DiffusersImage/README.md#attention-arguments)              |
| Choose, edit and reuse FLUX Blocks                                       | [Using FLUX Blocks](../modules/DiffusersImage/README.md#using-flux-blocks)                  |
| Implement the shared Cluster/User Node composite contract and V2 schemas | [Unified composite-node contract](unified-composite-node-implementation-plan-2026-09-01.md) |
| Track the first-party visual Diffusers/Transformers node system          | [Hugging Face visual node system plan](hugging-face-visual-node-system-plan.md)             |
| Track Diffusers, Modular Diffusers, speech, testing, and asset work      | [Hugging Face integration roadmap](hugging-face-integration-roadmap.md)                     |
| Review the Hugging Face-derived engineering and runtime requirements     | [Hugging Face engineering alignment](hugging-face-standards.md)                             |
| Review inherited source baselines and per-file modification notices      | [Source provenance map](source-provenance.md)                                               |
| Contribute code, nodes, dependencies, or client-facing changes           | [Contributing](../CONTRIBUTING.md)                                                          |
| Understand the local-only trust boundary or report a vulnerability       | [Security policy](../SECURITY.md)                                                           |
| Understand expected conduct in project spaces                            | [Code of conduct](../CODE_OF_CONDUCT.md)                                                    |

Custom Python and Hub block authors: [Developing custom nodes](custom-nodes.md),
including single-file drag/drop, automatic discovery and intentional Add/Load/Reload.

## Required Engineering Procedure

The [image prototyping readiness plan](image-prototyping-readiness-plan.md)
controls the next image handoff: audited model coverage, editable native stages,
model-change preservation, custom nodes, UX fixes and integrated qualification.
It records agreed scope and planned work, not completed implementation.

The [Creator / Developer workspaces plan](creator-developer-workspaces-plan.md)
is historical. The current editor has no audience-mode switch; custom sources
use one intentional Add/Load/Reload action. Older execution evidence is retained.

The [generic Diffusers workbench plan](generic-diffusers-workbench-plan.md)
defines the staged Auto/Expert authoring redesign, compatibility requirements,
and acceptance criteria. Planned behavior is not a current support claim.

Read [Cluster engineering lessons](cluster-engineering-lessons.md) before
node/Block, hierarchy, execution, qualification or cross-machine integration work.
The [runtime support matrix](runtime-support-matrix.md#model-families-and-support-boundaries)
summarizes family boundaries and separates application, model-quality, resource,
migration, and publication acceptance. Captured inputs and encoded output
measurements belong in the [API reference](api-reference.md#studio-preview-state);
monitoring and request latency belong in [troubleshooting](troubleshooting.md#resource-monitoring-during-execution).

## Documentation Standards

Public documentation should describe the supported current behavior and make its proof level clear. Keep commands executable from the directory stated, keep route/config/dependency names synchronized with code, and distinguish static or mocked validation from a live backend or real model-generation result.

Do not publish credentials, private media, personal paths, machine inventories, unredacted provenance, or dated internal execution trackers. Document supported product identifiers and contracts exactly as they appear in the current implementation.

When behavior changes, update the root README and the narrow guide in the same contribution. Verify repository-relative links and run the validation described in [CONTRIBUTING.md](../CONTRIBUTING.md) before requesting review.

- [Developer setup with uv and npm](developer-setup.md)
- [Service prototyping with API graphs](service-prototyping.md)

- [Workflow authoring and model selection](workflow-authoring-ux.md)
