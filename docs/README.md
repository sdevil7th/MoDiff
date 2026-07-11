# MoDiff Backend Documentation

This directory contains the durable technical guides for the MoDiff backend. Start with the root [README](../README.md) for installation and first launch, then use the guides below for the area you are working on.

## Read By Goal

| Goal                                                                        | Guide                                                                            |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Install, configure, launch, or update MoDiff                                | [Project README](../README.md) and [`config.example.ini`](../config.example.ini) |
| Understand HTTP and WebSocket surfaces                                      | [API reference](api-reference.md)                                                |
| Diagnose startup, ports, devices, downloads, media, or stale UI             | [Troubleshooting](troubleshooting.md)                                            |
| Understand the `modiff` namespace and retained `mellon` compatibility shims | [Backend namespace](modiff-backend-namespace.md)                                 |
| Build Modular Diffusers graphs and understand experimental compatibility    | [Modular Diffusers guide](../modules/ModularDiffusers/README.md)                 |
| Contribute code, nodes, dependencies, or client-facing changes              | [Contributing](../CONTRIBUTING.md)                                               |
| Understand the local-only trust boundary or report a vulnerability          | [Security policy](../SECURITY.md)                                                |
| Understand expected conduct in project spaces                               | [Code of conduct](../CODE_OF_CONDUCT.md)                                         |

## Documentation Standards

Public documentation should describe the supported current behavior and make its proof level clear. Keep commands executable from the directory stated, keep route/config/dependency names synchronized with code, and distinguish static or mocked validation from a live backend or real model-generation result.

Do not publish credentials, private media, personal paths, machine inventories, unredacted provenance, or dated internal execution trackers. Compatibility names retained for existing graphs, clients, imports, or upstream APIs should be explained instead of mechanically renamed.

When behavior changes, update the root README and the narrow guide in the same contribution. Verify repository-relative links and run the validation described in [CONTRIBUTING.md](../CONTRIBUTING.md) before requesting review.
