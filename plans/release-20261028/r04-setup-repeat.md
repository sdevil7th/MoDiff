# R04 setup and repeated-generation checkpoint

The setup repair is delivered and the bounded Linux execution checks are
verified. R04 remains open for the external and capability-specific rows below.
R05 and R06 retain their own completion criteria.

The tested implementation is backend
`e06a4f1615e2b675834b1dde1ac552054e675095` and client
`c568d342287fd79bdf6ffa67e34ff1c03b9fd3eb`. The client product remains byte-identical
to `f6cddacc57e799e4daf259a4e01ffb970d57a99a`; its R04 changes update documentation
and the immutable backend references used by CI. These changes remain in the
existing backend and client PR7s.

## Setup repair

Missing, unreadable or invalid installed package-version metadata previously
could escape readiness inspection. The base-runtime observation now names the
affected required package and returns the existing repair guidance. Valid
versions outside the supported range remain a separate issue. An installed
Diffusers base with damaged metadata cannot gain readiness through a provider's
reported version.

The observation preserves the existing accelerator/profile repair route. It
does not install packages, change the environment or import a model. Unexpected
resolver programming failures remain visible. Transformers and PEFT remain
ordinary initial-setup dependencies; this change adds no optional installation
or activation step.

The [setup and recovery guide](../../docs/developer-setup.md#workflow-setup-checklist)
documents the resulting behavior. Reuse the closed R02 selected-scope checklist,
installation/verification/activation states and native status-recovery evidence
at their recorded source scope.

## Software verification

| Check | Actual result |
| --- | --- |
| Original damaged-metadata regression | Real temporary distribution metadata retained 17 failures and two passes on the previous source. The first fixture mistake and corrected original failure remain recorded separately. |
| Related readiness/setup tests | 210 passed, 46 subtests, one expected Windows POSIX-permission skip and five recorded warnings; 45.25 seconds. |
| Complete backend gate | Ruff, dependency verification and preflight passed; 9140 tests and 13202 subtests passed, 761 skipped and 55 warnings; pytest took 2209.92 seconds. |
| Complete compatible client gate | `npm run check` passed in 798.46 seconds; all 119 product assets match the packaged client exactly. |
| Hosted CI at the tested heads | Both PRs had all four jobs passing; all 16 PR/push jobs passed, observed 7 November 2026 at 22:06:47 UTC. Later documentation-only heads need their own honest CI observation. |
| Protected delivery | Tested backend and unchanged client product were adopted on the configured Linux host, preserving the existing data, media, environment, configuration, cached artifacts and 119 disk/118 served assets. |

The recorded commands below use `python` for the configured reviewed test
interpreter. Original receipts retain its exact absolute path and environment
bindings. The related test invocation was:

```text
python -X utf8 -B -m pytest -q -rs -p no:cacheprovider tests/test_base_runtime_metadata.py tests/test_native_uv_runtime.py tests/test_runtime_environment.py tests/test_runtime_profile.py tests/test_install_guidance.py tests/test_install_detection.py tests/test_install_toolchains.py tests/test_workflow_setup.py tests/test_optional_runtime_server.py tests/test_diffusers_stable_distribution.py
```

The complete backend gate used the documented reviewed-source environment and
these command forms:

```text
uvx --offline ruff==0.12.7 check . --select E9,F
uv pip check --python python
python -X utf8 -B -m modiff.preflight --json --check-port 8088 --fail-on-error
python -X utf8 -B -m pytest -q
```

These are software and CPU/bootstrap checks. The three hosted OS setup jobs do
not establish fresh accelerator installation or physical GPU qualification.
Reuse the closed gates for unchanged executable source; documentation and
private evidence readers do not justify repeating them or deploying again.

## Configured Linux execution

Nine completed ordinary UI runs used one immutable cached Z-Image Turbo revision and one
compatible immutable LoRA artifact. The graph contained seven ordinary nodes
and 69 actual form fields. Every run kept 1024×1024 output, eight inference
steps, CFG 1, unquantized BF16, automatic attention and whole-workflow Auto.
Only the declared seed, prompt and adapter-scale edits changed authored inputs.

| Observation | Native seconds | Evidence and disposition |
| --- | ---: | --- |
| First cold generation, seed 4210 | 69.132 | Actual loader/encoding/denoising execution and retained output. |
| Unchanged generation | 0.782 | All seven actual node events reported cached results; decoded pixels were identical. This was result reuse. |
| Seed 4211 | 52.040 | Loader and prompt stayed cached; denoising executed and pixels changed. |
| Changed glass-color prompt | 52.475 | Prompt encoding and denoising executed; the loader stayed cached and pixels changed. |
| Adapter scale 0 at fixed seed/prompt | 52.911 | Actual activation projection recorded scale 0 and pixels changed. |
| New cold zero-scale owner | 59.558 | A separately admitted cold loading path established the starting owner for the unrun warm-restoration check. Its pixels exactly matched the earlier zero-scale output. |
| Warm restoration to scale 0.8 | 54.114 | Actual activation and state-hash transition; pixels changed from scale 0 and exactly matched the earlier 0.8 state. |
| Ordinary Recompute | 53.552 | The outputs-only action retained the loader and prompt cache; denoising executed. Decoded pixels exactly matched the restored state. |
| Scoped release followed by cold generation | 61.497 | The cache was empty before entry; real loading and denoising executed. Decoded pixels exactly matched the corresponding warm state. |

Each result joins the actual task, attempt, session, workflow, source, dispatch,
consumed inputs, durable media and served/displayed image. Same-state comparisons
kept the predeclared PSNR ≥30 and MAE ≤4 bounds; changed-input comparisons
required different decoded pixels. All measured same-state comparisons above
were exact. Raw types remain recorded: ordinary editing stored seed 4211 as a
string, and the existing Denoise normalization converts it to an integer before
creating the generator. Task-level form metadata and edited node inputs remain
separate original records.

The test guards required 100 GiB available RAM at cold entry, a 20 GiB reserve,
75 GB free disk, at most 25 GB scratch, at most 1 GiB owned-worker swap growth,
at least 2 GiB free swap, pressure checks and a finite 30-minute scope. These are
test-admission safeguards, not published minimum hardware requirements.
Observed owned swap growth and scratch allocation were zero; sampled available
RAM stayed at or above 89,047,035,904 bytes.

An initial private monitor cancelled a zero-output attempt before model loading
because it assumed a full runtime-fingerprint object was already available.
The original error, supported cancellation and recovery remain retained.
Completed outputs were not rerendered to repair readers. Other retained reader
issues involved display-only preview URLs, inherited-reference locations and
optional frontend media bookkeeping. Their corrections preserved original
receipts and strict source/control/byte/pixel checks.

The first 30-minute monitoring window ended while idle after five completed
outputs; it cancelled no model task. That scope remains incomplete. The four
lifecycle runs used a separately declared independent scope, completed in
712.27 seconds including observation and cleanup. There is no claim that all
eight originally proposed behaviors completed in one 30-minute series.

Normal owned releases and QA closure passed. All prior 94 output records,
88 unrelated preview slots and original 412 workflow bodies remain preserved;
the final store contains 103 outputs, 89 slots, 682 files and 423 workflow
bodies. New owned documents and edits are bound to genuine UI saves. Both
durable media aliases, unrelated work, the original idle worker, source,
runtime, packages and configuration remain preserved. The current cache is
empty, with no scratch or active download. No package installation, model
download, cache move, OS tuning or public-media upload occurred.

## Remaining acceptance boundaries

| Row | State |
| --- | --- |
| Original 16 GB GPU/32 GB RAM Windows complaint | Externally held under the Linux-only pretrained-model rule. Large Linux GPU success and CPU CI do not close it. |
| Fresh GPU installation and optional-runtime matrix | Not established by an already configured machine or temporary metadata tests. Existing setup contracts and CPU bootstrap evidence retain their narrower scope. |
| Different trained model/adapter artifact replacement | Native proof remains held. Scale changes of one cached artifact do not establish a second artifact's replacement. No additional download is authorized. |
| Formal cached resident-adapter identity | Held. For ordinary model-CPU ownership the reporter intentionally returns unavailable ownership telemetry; its strict reusable seal requires a supported native policy. Actual cache events, activation records and pixels are retained without inventing an application ID or resident-state seal. |
| 43 missing-model cases | User-held; no downloads, model moves or SSD configuration changes. |
| Broader release | R05 transport attribution and R06 creative, access, capability, hardware, provenance, template and publication decisions remain separate. |

Original receipts, private helpers, failures and generated media remain outside
Git. The existing same-PR descriptions and private R04 review checkpoint index
record their exact source scopes. No human visual verdict, public qualification,
publication or merge approval follows from these execution tests.
