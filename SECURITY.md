# Security Policy

## Supported security boundary

MoDiff is early-stage, local-first software. Only the current `main` branch is maintained, and it is not presented as a hardened production or multi-user service.

The supported deployment boundary is one trusted user running the backend on a trusted machine with the default `127.0.0.1` bind address. The server does not provide authentication, authorization, tenant isolation, request signing, or a Python/model sandbox. CORS and TLS configuration do not add those missing controls.

Do not expose MoDiff directly to an untrusted LAN, the public internet, a shared workstation population, or untrusted browser origins. A reverse proxy is not sufficient unless it also supplies strong authentication, authorization, origin controls, request-size limits, and network policy for every HTTP and WebSocket route.

## Code-execution risks

MoDiff is designed to execute Python and model code:

- Custom-module installation can clone a Git repository or copy a local directory into `custom/`, then import it into the live registry.
- Reviewed model-execution libraries maintained by Hugging Face run in the backend process with the same filesystem, network, CPU, and accelerator access as MoDiff. Official maintenance reduces neither package supply-chain risk nor the need to review the selected version and integration.
- Repository-supplied Python would run with backend-process permissions. Current custom Modular Diffusers paths are `contract_only` and reject `trust_remote_code` before model construction; exact cached 40-character commits are still required for Hub contract preview. Do not weaken that fail-closed boundary or accept moving branches/tags if executable support is added later.
- Model deserialization and optional native/CUDA packages have their own supply-chain and memory-safety risks.
- Workflows can allocate substantial CPU, RAM, accelerator memory, disk, and network bandwidth.

Install only sources you trust. Review repository ownership, code, dependencies, model licenses, and the exact revision before installation. Prefer immutable commit revisions over moving branches. Disabling a module after import does not undo code that has already run; restart the backend after changing trusted code.

A package being part of the Hugging Face ecosystem is distinct from a model being hosted on the Hub. Do not treat a Hub namespace, model card, or `trust_remote_code` implementation as first-party library code. Optional model runtimes, including Transformers, require an explicit local install action and version verification; template browsing, registry discovery, and Auto planning must remain non-installing operations.

## Tokens and secrets

The Models UI can validate a Hugging Face token and writes it to `[huggingface] token` in `config.ini`. That file is ignored by Git, but the token is plaintext and is not protected by an operating-system credential store.

- Use a least-privilege Hugging Face read token.
- Never commit or share `config.ini`, `.env` files, private keys, or authenticated URLs.
- Rotate a token immediately if it appears in logs, screenshots, workflow packages, shell history, or a commit.
- Review a staged root commit with a secret scanner before publication; rewriting visible Git history does not automatically erase local reflogs or unreachable objects.

## Files, outputs, and privacy

The configured `work_dir` is the intended file-browser boundary. The backend can read, upload, preview, and write files within configured paths. Keep `work_dir`, `data`, model caches, and temporary/offload directories narrow and dedicated to MoDiff. Do not point them at a home directory, source tree, credential store, or shared network volume unless that access is intentional.

Generated media, prompts, graph snapshots, Studio output history, reusable blocks, and workflow shares are stored as local plaintext under `data/` by default. They may contain sensitive input media, prompt text, model identifiers, or execution metadata. MoDiff does not provide encrypted storage, automatic retention policy, or secure deletion.

## Network behavior

Depending on the workflow, MoDiff can connect to Hugging Face, Git hosts, model-defined URLs, and other user-supplied sources. Backend-managed web-media import rejects credentials and non-public address resolutions, validates every redirect, disables environment proxies for graph-controlled URLs, and connects to the validated numeric address to resist DNS rebinding. Installed Python/model code and specialized downloaders still have process-level network access. Use host firewall and egress controls when running untrusted or sensitive workloads. Offline mode reduces Hugging Face resolution but does not turn arbitrary installed code into a sandbox.

The main HTTP server defaults to a 1 GiB request cap, workflow-share preview copies are capped at 256 MiB, and every HTTP request requires a loopback request Host (`localhost` or a literal loopback address) and loopback peer. Browser requests that supply an Origin must use a loopback HTTP(S) Origin; originless native clients and ordinary browser navigations are accepted only from loopback. The supervisor control plane is loopback-only. WebSocket upgrades also require a loopback destination and peer. Browser WebSockets must provide a loopback `http` or `https` Origin; native clients without an `Origin` header are accepted only over a loopback connection. Initial WebSocket history contains compact task receipts, while completed workflow snapshots remain available lazily through `/runs/{task_id}`. These controls resist ordinary cross-site requests, read-side data exposure, and DNS-rebinding hostnames, but they do not add authentication or make a non-loopback deployment supported.

## Reporting a vulnerability

Use the repository's private **Report a vulnerability** or security-advisory flow when it is available. If no private reporting channel is configured, open a minimal public issue asking the maintainers to establish private contact. Do not include the vulnerability, exploit details, credentials, private media, tokens, local paths, or host information in that issue.

Once a private channel is established, include:

- The affected commit and platform.
- Reproduction steps or a minimal proof of concept.
- Expected impact and whether code execution, file access, token disclosure, or cross-origin access is involved.
- Any suggested mitigation.

Do not include real access tokens, private model contents, personal media, or destructive payloads in a report. Please avoid publicly disclosing an unpatched vulnerability until maintainers have had a reasonable opportunity to investigate and release guidance.
