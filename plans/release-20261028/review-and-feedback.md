# Reviewed baseline, chronological feedback and remaining gaps

Read this with the [deadline and ownership plan](README.md). Findings describe the
reviewed BE `ae09c201` / FE `fd3d3dac` pair. They distinguish missing implementation,
diagnosed defects, incomplete native acceptance and externally blocked work.

## Clean-baseline review completed before planning

The review inspected Git status, staged/unstaged/untracked paths, worktree heads,
history, preserved integration records, current authored changes, their call paths
and tests, and live remote branch identities. It did not change the application,
install packages, run models, restart services or perform another full test campaign.

| Scope                                        | Result                                                                                                                                      |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Backend                                      | 12 worktrees: 10 clean, two historical dirty overlays                                                                                       |
| Client                                       | 15 worktrees: 11 clean, four historical dirty overlays                                                                                      |
| Active primary and paired delivery checkouts | Clean, exact reviewed pair                                                                                                                  |
| Deployed Windows source checkout             | Clean at reviewed backend commit                                                                                                            |
| Historical backend overlays                  | 272 + 197 changed paths; 461 surviving file bodies and eight deletions match the preserved integration snapshot; statuses/indexes unchanged |
| Historical client overlays                   | 106 changed paths; exact preserved bytes/status path sets, zero newly added/missing paths or drift                                          |
| Integration status                           | Existing source incorporated or superseded; no new pending implementation found                                                             |

The backend historical overlays include 133 formerly untracked files; the client
image/audio overlay includes 17. These are preserved historical work, not a reason
to bulk-stage old files. Current client has all 17 and all 61 added literal test
titles extracted from that overlay's modified tests. Fifty of its 100 files still
match current contents after normalization; fifty have later primary changes.
The historical 54-exact/33-incorporated/13-superseded classification describes the
earlier integration, not today's whole-file equality.

Other client overlays retain the already integrated policy/hydration fixes and
an unfinished Auto browser test whose intended coverage is superseded by current
FP8r/NVFP4 accepted/refused cases. The backend merge
`93fe5949e528f698a1cbed60bf661a9d0d9cc859` is an ancestor of the reviewed head.
One old backend contract branch has patch-equivalent commits rather than a new
missing implementation. Preserve all original copies and private records. A clean
new starting pair does not require destroying historical worktrees.

Latest authored BE review covered `modiff/applied_adapter_provenance.py`,
`execution_input_provenance.py`, `native_disk_lora.py`, `NodeBase.py`, the affected
Diffusers loaders/call-input producers and their regressions. FE review covered
`resolvedExecutionInputs.ts`, `applied-adapter-provenance.mjs`, the provenance
factory, canonical recorder/reader, their negative tests and cache-release UI.
No new high-confidence runtime defect was established. Optional incomplete
evidence remains incomplete; it is not falsely promoted to acceptance.

Both working-branch remote heads were updated normally to the reviewed application
pair before these plan files were written. The historical overlays were neither
reapplied nor deleted. No public media was published.

## Chronology: what changed, and what did not become accepted

Dates below are the supplied feedback dates. Implementation dates and commits are
separate. A commit subject or a developer announcement is not a native pass or
the reviewer's approval.

| Feedback                                                                                            | Implementation/history inspected                                                                                                                                                                                                                                   | Current disposition and task                                                                                                                                                                                                            |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 20 August: quick iteration should be simpler than coding; one magic node plus editable stages       | BE `e0481ed2` had generic contracts. September 12 BE `d70d737f` / FE `5c0b755` added the later nested modular representation and persistence. Current `BlockNodeV2.tsx`, `blockPersistenceV2.ts` and backend composition contracts implement the actual hierarchy. | Implemented for reviewed native routes. Continuous model-backed usability/persistence still needs W4/L1. Arbitrary imported Python is not automatically decompiled into editable stages.                                                |
| 20 August: import custom Blocks from Hub and remember models/parameters                             | September custom-component loading includes BE `a1f24642`, `4bc0028f`; FE September 26 `9378cc7` simplifies intentional Add/Load. Current custom-extension registry and UI are implemented.                                                                        | Missing integrated current-host evidence, not a missing Add button. L1 plus W4. Add/Load/Reload is explicit code authorization; graph import is not.                                                                                    |
| August follow-up: edited Cluster becomes reusable user node                                         | Current accepted persistence contract stores workflow-instance edits and explicitly snapshots a reusable Block with Save as User Node.                                                                                                                             | The original automatic conversion proposal was superseded intentionally. Do not reintroduce destructive automatic conversion as a feedback fix.                                                                                         |
| 11 September: four understandable stages, visible graph, compatible family/task changes             | FE `c08b02b` generic authoring, `7477c8b` model-change mapping; BE `59bc3421` task-first authoring; FE `c7af715` later visual grouping. Current `operationAuthoring.ts`, stage grouping and model/task review/Undo paths are implemented.                          | W4 covers a representative distinct transition, existing custom wiring and restart. Exactly four boxes everywhere is not a requirement; meaningful Guidance/Preview/input nodes may remain. All-family transitions remain broader work. |
| 11 September: typed suggestions when dragging from an empty port                                    | Current `NodeSearchDialog.tsx`, `nodeConnectionMatching.ts` and type-compatibility logic implement verified matches plus explicit unverified matches.                                                                                                              | Implemented; include the actual gesture in W4. Nominal type equality alone cannot establish arbitrary tensor semantics.                                                                                                                 |
| 11 September: repeated load/encode/unload/crash; Qwen would not run on Windows 16 GB/32 GB          | BE September 17 `897e27a9` separates recompute/model lifetime; `b2e8ec5a` strengthens reuse; September 26 `01d91919` fixes cached Modular loading. Later owner/accounting fixes remain in current code.                                                            | Exact current Windows original-route acceptance remains open: W3. Historical larger/shared-memory Qwen output cannot close it.                                                                                                          |
| 26 September: announcement of generic image workflows, exposed controls and uv/npm/custom-node docs | September 26 `e7cb7353`, `a0d29db8`, FE `9378cc7` and existing workbench guides.                                                                                                                                                                                   | The reply promised a later test, not approval. Assess actual simplicity through W4, not port/node counts.                                                                                                                               |
| 5 October: setup, core dependencies, Windows platform behavior, progress and memory failures        | BE October 7 `0183fbed`, October 8 `11db2aa9`, October 9 `30924930`, `283c55b9`, `c43b5709`, `1634d936`; FE optional-runtime recovery `e2f535d` and later transport fixes.                                                                                         | Most implementation exists; original native installation/repeat-run acceptance remains W2/W3/W5. Details below.                                                                                                                         |
| 26–27 October: consolidation and current Windows update                                             | BE `93fe5949`, `9d991969`, `a099c7b5`, `ae09c201`; FE `758d3564`, `fd3d3dac`.                                                                                                                                                                                      | Current tested build exists. Saved-list acknowledgment and cache-release timeout/error fixes are already delivered; do not schedule them as unimplemented.                                                                              |

## Each 5 October item

| Item                                           | Actual current code/contract                                                                                                                                 | Remaining action                                                                                                                                                                         |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. No-script path still called scripts         | `BE/docs/developer-setup.md`, `pyproject.toml`, native `uv sync`/`uv run` and FE `npm ci`/`npm run dev` are direct.                                          | W1 fixes stale distribution target; W2 executes the ordinary Windows instructions from a new isolated environment.                                                                       |
| 2. Redundant flags                             | Ordinary examples no longer need `--no-sync`/`--no-project`. Isolated managed bootstrap legitimately has a different context.                                | Check the actual quick start; do not remove meaningful bootstrap isolation globally.                                                                                                     |
| 3. Allocator and missing Triton                | `BE/modiff/runtime_environment.py` applies platform policy before imports; `runtime_compilation.py` requires a real optional kernel probe.                   | W2 proves eager Windows generation with Triton absent; optional compiler support remains separate. Preserve operator overrides.                                                          |
| 4. Transformers optional                       | Required base dependency in `BE/pyproject.toml`; `modiff/base_runtime.py` verifies readiness lazily.                                                         | W2 fresh installation and isolated repair. No product dependency redesign needed.                                                                                                        |
| 5. PEFT optional                               | Required base dependency; ordinary adapter integration uses upstream APIs.                                                                                   | W2 base readiness, W3/W4 ordinary Windows LoRA when its route/artifact is available, L2 exact Linux adapter controls. A tiny or Linux test does not prove full Windows LoRA replacement. |
| 6. uv pinned/no upgrade guidance               | Operator uv accepted; tool upgrade, project lock update and ABI changes are documented separately.                                                           | W2 verifies commands in a disposable environment; no broad dependency upgrade.                                                                                                           |
| 7. ECONNABORTED spam                           | FE `scripts/backend-proxy-lifecycle.ts` narrowly handles owned disconnected Vite sockets; actual backend failures remain visible.                            | W5 production and normal Vite refresh. Capture an emitting stack if it recurs.                                                                                                           |
| 8. Immediate missing-package UI error          | Base-included runtime profiles suppress optional Install/Activate; damaged foundations expose repair.                                                        | W2 verifies healthy startup and isolated damaged-environment recovery. Never damage the user's live runtime.                                                                             |
| 9. Install/Activate delay and unclear status   | FE `RuntimeOptimizationsCard.tsx`, `optionalRuntimePolling.ts`; BE durable optional job identity and ready replacement-worker checks.                        | W5 current visible progress/recovery. Actual fresh optional install is conditional and separate from ordinary base generation.                                                           |
| 10. Second-run Combined systemRamBytes failure | `BE/modiff/workflow_auto_resource.py` separates RAM/VRAM, grants exact resident credit once; `workflow_auto_lifecycle.py` checks actual post-release memory. | W3 original cold/warm sequence. The message described host RAM, not dedicated VRAM.                                                                                                      |

## Concrete gaps, with proof level

1. **Wrong distribution instructions — diagnosed.** BE README and developer
   setup, FE README/CONTRIBUTING/Windows links still name `fix-ui-ux-issues`.
   Remote heads at review were BE `1634d936`, FE `2146f456`, behind the tested pair.
   Current working/delivery branches contain the pair. W1 fixes the instructions
   and their command-contract assertions; it must not silently merge public main.
2. **Windows Qwen 16 GB VRAM/32 GB RAM — native acceptance missing.** No exact
   current qualifying receipt was established. Source tests and historical Qwen
   runs on other hardware do not establish fit. W3 may discover a real defect or
   an honest capacity limit; neither outcome should be guessed in advance.
3. **FLUX3 W01 — genuine retained cold-load capacity hold.** The latest retained
   load needed 14,850,133,600 bytes of host RAM with 12,097,413,120 available.
   Availability fell 3,298,099,200 bytes after dispatch; causality is unproved.
   This is distinct from Qwen and the old warm-run complaint. No current duplicate
   reserve bug was found. Do not lower the guard or close arbitrary user apps.
4. **Continuous custom/model/service journey — acceptance gap.** The pieces
   exist. Existing model-free service export and representative model-switch
   results do not close the same edited, saved, restarted, model-backed journey.
5. **General canonical collector — implementation incomplete.** FE
   `scripts/release-contract-core.mjs` rejects deterministic builtin qualification
   and input-backed canonical evidence. Current accounting join supports a narrow
   input-free DDIM path. Actual native smoke had three grouped nodes rather than
   the five-node canonical fixture, unaccounted loader profile/recipe fields and
   an image-contract metadata mismatch. Do not edit original receipts to fit it.
6. **Native smoke display versus media — retained discrepancy.** Durable output
   is correctly 32×32 while legacy top-level/form dimensions remained 1024×1024.
   Trace requested/form/consumed/encoded origins before any metadata/UI fix; this
   does not prove that the model generated at the requested displayed size.
7. **Strict applied-adapter multi-owner acceptance — explicit narrow scope.**
   FE `scripts/applied-adapter-provenance.mjs` strict mode requires all executed
   loader owners while individual proof records are output-ancestry scoped. The
   existing strict joint/reverse protocol uses one owner; ordinary optional
   recording is unaffected. Extend only with a deliberate multi-owner contract
   and failing regression when that broader claim is needed.
8. **Historical first-click issue — unresolved, not reproduced as current.**
   `docs/workflow-authoring-ux.md` retains an early non-submission after switching;
   later repetitions passed. W4 observes first-click behavior; do not invent a fix.

## Existing proof that should be reused honestly

The exact pair has full software gates and one genuine 32×32 DDIM Custom/Expert
image, plus save, close, reopen, scoped cache release and original-data preservation.
Its Auto candidate is null. It proves current deployment/execution at that scope;
it does not qualify Auto, repeated inference, full-weight LoRA, the unchanged
canonical graph, every workflow or any absent hardware.

Linux has 12 retained finite executions: ten original cases and two scale-zero
controls. Six original outputs are visual candidates; four remain quality holds.
The missing joint/reverse controls are L2. Historical disk cleanup resolved the
old capacity floor with a narrow margin; fresh admission is still required. Do
not repeat completed cases unless source impact or a new hypothesis warrants it.

## Mistakes the next execution must avoid

| Observed failure pattern                                                                           | Required prevention                                                                                                     |
| -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Repeated full suites consumed implementation time                                                  | Start with a reproduction and focused checks; reserve one final stable-source gate cycle.                               |
| Wrong base versus evaluated optional environment produced hundreds of test failures                | Resolve interpreter/profile first. Keep stable base checks and validated optional-runtime checks separate.              |
| A Windows `.venv` junction resolved into another checkout                                          | Resolve real environment paths before sync/install; isolated directory names alone do not establish isolation.          |
| Tests constructed a supervisor against live data                                                   | Set isolated storage before importing the server singleton; assert queue/data roots.                                    |
| Sealed optional environments gained bytecode through bare PYTHONPATH                               | Use `scripts/test_reviewed_optional_runtime.py` and suppress bytecode writes through child processes.                   |
| Source, HMR or collector changed during a run                                                      | Freeze the actual serving and evidence-producing bytes, not merely a Git label.                                         |
| Old form/model identity substituted for actual execution                                           | Join original task, graph, model owner, consumed inputs and durable media; keep discrepancies visible.                  |
| Cache hit reported as independent inference                                                        | Inspect per-node compute/cache observations and timings; seed/prompt changes must cause appropriate computation.        |
| Longer timeout hid unresponsive cleanup                                                            | Profile HTTP/worker responsiveness and lease ownership; 120 seconds is only the full release request deadline.          |
| UI test hit a toast/overlay or stale node geometry                                                 | Inspect actual pointer target/drag threshold; preserve the failed harness evidence and repeat only the missing gesture. |
| Missing PATH/USERNAME, fonts, sibling checkout or artifact capacity caused repeated setup failures | Verify these prerequisites once in the actual isolated runner; two same-category failures trigger diagnosis.            |
| Old resource or Gallery counts treated as new truth                                                | Refresh the denominator at source freeze; compare semantics and retain original scope.                                  |
| Output media/source inputs or derivative videos were confused                                      | Select the declared output and compare task/hash; decoded-frame headers can differ without established pixel drift.     |
| Quality failure was treated as a receipt or rerun problem                                          | Preserve its failed visual criterion; after two failures require a changed hypothesis.                                  |
| Merge/push described as deployed/accepted                                                          | Verify running source and served bundle, then actual affected user behavior.                                            |

The mirrored `docs/cluster-engineering-lessons.md`, CONTRIBUTING, SECURITY and
workbench acceptance guide remain required reading. This plan supplements them;
it does not authorize changing quality criteria or weakening execution boundaries.
