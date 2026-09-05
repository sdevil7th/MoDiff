# MoDiff Backend Documentation

This directory contains the durable technical guides for the MoDiff backend. Start with the root [README](../README.md) for installation and first launch, then use the guides below for the area you are working on.

## Read By Goal

| Goal                                                                        | Guide                                                                            |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Install, verify, run a first workflow, configure, or update MoDiff           | [Project README](../README.md) and [`config.example.ini`](../config.example.ini) |
| Understand HTTP and WebSocket surfaces                                      | [API reference](api-reference.md)                                                |
| Diagnose startup, ports, slow/stalled runs, devices, downloads, or media    | [Troubleshooting](troubleshooting.md)                                            |
| Compare the qualified accelerator profiles and their proof levels           | [Runtime support matrix](runtime-support-matrix.md)                              |
| Check quantized model, dependency, download, and qualification support       | [Quantization support matrix](quantization-support.md)                           |
| Review optional attention, quantization, and compilation capabilities        | [Optional runtime optimizations](optional-runtime-optimizations.md)              |
| Build Modular Diffusers graphs and understand experimental compatibility    | [Modular Diffusers guide](../modules/ModularDiffusers/README.md)                 |
| Implement the shared Cluster/User Node composite contract and V2 schemas     | [Unified composite-node contract](unified-composite-node-implementation-plan-2026-09-01.md) |
| Track the first-party visual Diffusers/Transformers node system              | [Hugging Face visual node system plan](hugging-face-visual-node-system-plan.md)  |
| Track Diffusers, Modular Diffusers, speech, testing, and asset work          | [Hugging Face integration roadmap](hugging-face-integration-roadmap.md)          |
| Review the Hugging Face-derived engineering and runtime requirements        | [Hugging Face engineering alignment](hugging-face-standards.md)                  |
| Review inherited source baselines and per-file modification notices         | [Source provenance map](source-provenance.md)                                    |
| Contribute code, nodes, dependencies, or client-facing changes              | [Contributing](../CONTRIBUTING.md)                                               |
| Understand the local-only trust boundary or report a vulnerability          | [Security policy](../SECURITY.md)                                                |
| Understand expected conduct in project spaces                               | [Code of conduct](../CODE_OF_CONDUCT.md)                                         |

## Documentation Standards

Public documentation should describe the supported current behavior and make its proof level clear. Keep commands executable from the directory stated, keep route/config/dependency names synchronized with code, and distinguish static or mocked validation from a live backend or real model-generation result.

Do not publish credentials, private media, personal paths, machine inventories, unredacted provenance, or dated internal execution trackers. Document supported product identifiers and contracts exactly as they appear in the current implementation.

When behavior changes, update the root README and the narrow guide in the same contribution. Verify repository-relative links and run the validation described in [CONTRIBUTING.md](../CONTRIBUTING.md) before requesting review.
