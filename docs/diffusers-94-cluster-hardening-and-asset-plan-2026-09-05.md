# Diffusers 94-Cluster Hardening and Asset Qualification Plan

Date: 2026-09-05

Status: active implementation plan

## Current guidance

This dated plan is historical context, not current route or release approval.
Use the [runtime support matrix](runtime-support-matrix.md#model-families-and-support-boundaries)
for family boundaries, the [engineering procedure](cluster-engineering-lessons.md)
for verification, and [runtime troubleshooting](troubleshooting.md#resource-monitoring-during-execution)
for resource monitoring and request responsiveness.

## Immediate delivery plan — Qwen closure, then cached FLUX

Updated 2026-09-05 after the overnight regression checkpoint. This sequence
takes priority over further catalog expansion and publication work below.

| Order | Deliverable and acceptance | Estimate from this checkpoint |
| --- | --- | --- |
| 1 | Diagnose and fix Qwen browser failures: nested-library navigation and dynamic compiler skeleton ownership. Preserve the production ownership guards. Focused browser tests must pass against the current UI and compiler. | 1–2 hours |
| 2 | Close the current Qwen editing lifecycle: parameter locality; nested expansion/resize/links; compatible replacement and reconnection; nested User Node save/reuse; persistence and instance isolation; actual frontend generation from the modified workflow. Reuse existing evidence only where its definition and tested behavior remain applicable. | 1–2 additional hours, including model execution |
| 3 | Finish client/backend checks and the shared browser gate. Regenerate dependent source ledgers in dependency order; investigate failures instead of changing expected results without a cause. Record exact passed, failed, and skipped cases. | Included in the Qwen 2–4-hour target where checks can run alongside independent work |
| 4 | Demonstrate cached FLUX.2 Klein 4B text-to-image through the visible frontend: insert, edit, expand/resize, save/refresh, generate, view and persist media. Use official model recommendations and a complex prompt; preserve the exact workflow and parameters. | 2–4 hours after Qwen closure |
| 5 | Qualify the six cached FLUX routes: FLUX.2 Klein and Klein Base text-to-image/edit, plus Kontext text-to-image/edit. Exercise compatible structural changes and save/reuse behavior as well as generation. | 6–12 hours total for the FLUX tranche |

Qwen plus one demonstrated FLUX workflow is a 4–8-hour target, with another
working day of contingency if a browser failure exposes a product defect or
model execution exposes an upstream/runtime incompatibility. Report a missed
estimate and its cause immediately; successful isolated runs do not close the
integrated gate.

The current cache inventory contains the exact artifacts for those six FLUX
routes. FLUX.1-dev and FLUX.2-dev do not have complete required snapshots;
they follow the cached variants. Recheck Model Manager before any installation.
Downloads use the application's Hugging Face Hub pull exclusively.

New delivery evidence belongs under
`data/review/qwen-flux-delivery-2026-09-05/`, with separate Qwen and FLUX
subdirectories. Keep generated assets, screenshots, parameters, workflow
exports, and technical receipts together. Historical evidence remains in its
original folder and is linked explicitly rather than copied into the new batch.

Current checkpoint: all 11 Qwen admissions have recorded frontend generations.
The final backend suite now passes: 2,464 tests, 54 intentional skips, and 6,988
subtests. The complete client `check` also passes after all fixes below,
including the production build and bundle gate. The final `check:ui` passes:
two shared-control tests and 126/126 mocked Studio browser tests. Earlier
115/126 and 125/126 runs were intermediate failures, not completion evidence.
The protected-owner fixture needed all targets in the viewport and a small
pointer move across the drag threshold before its long move. The deployed
clean-browser Qwen smoke also passes. No model family is considered fully
qualified from the all-94 structural audit alone.

Implemented during this delivery tranche:

- The mocked catalog compiler now matches the production hidden compilation
  transaction, preserving ownership while its backend fields finalize.
- Browser assertions navigate the nested catalog and explicitly expand a
  nested loop; expanding the root must not expand all descendants.
- Subtree saves remove excluded public/control mirror targets and renumber
  retained controls contiguously without altering their relative order.
- All 94 workflows pass 1,700 independent subtree save/reinsert checks, with
  original-instance immutability. Receipt: `all-94-subtree-audit.json` in the
  new delivery evidence root.
- Current Qwen text-to-image passes its real FE lifecycle, including two-instance
  isolation, parameter locality, save/refresh, nested movement/containment,
  collapsed/expanded export equivalence, generation, save as User Node, and a
  second generation from that saved User Node. Evidence is under
  `qwen/qwen-image-2512-block-v2-live/`.
- Current Qwen compatible prompt-node replacement and delete/reconnect followed
  by save/refresh and real generation passes. Evidence is under
  `qwen/qwen-v2-structural-execution/`.
- Saving the actual nested Qwen text-encoder frame through its toolbar, adding
  that saved definition from User Nodes, independently editing its prompt, and
  saving/refreshing both instances passes against the live backend. Cross-boundary
  sockets and saved defaults survive. Evidence: `qwen/nested-subtree-reuse/`.
  This proves reuse/persistence, not standalone generation from an unconnected
  text encoder; that subtree still requires compatible pipeline components.
- The full visible replacement/adoption/reconnection/move-out/interface/undo-redo
  browser sequence passes. The corrected test verifies the actual dropped node
  center before releasing the mouse instead of assuming the requested pointer
  trajectory was applied.
- The clean-browser deployed smoke exposed an additional Arrange graph defect:
  the legacy arranger was independently resizing/repositioning V2 projections,
  competing with the recursive Block fitter. Global Arrange now moves V2 roots
  as whole units and durably saves their positions, leaving internal layout and
  semantics unchanged. Unit tests cover undo/rematerialization; the all-94
  hierarchy audit covers projection preservation after Arrange.
- Saving/reinserting a nested User Node exposed click insertion over an expanded
  root. User Nodes now use collision-free placement, including a rightmost-edge
  fallback when the bounded search is full, and focus the inserted node.
  Live nested-subtree reuse now asserts no overlap and passes.
- The post-layout-fix complete client `check` passes. That build is now served
  on port 8088, and a clean-browser production smoke passes recursive DOM
  containment/non-overlap checks, nested expansion/resizing, and Arrange.
  Screenshot: `qwen/deployed-expanded-qwen.png`. The final shared browser
  suite also passes; the first FLUX frontend execution has now started.
- The first FLUX.2 Klein V2 demo test is implemented using shared V2 lifecycle
  helpers. Its immutable model-card example recommends 1024×1024, four steps,
  guidance 1.0, BF16 and model CPU offload:
  <https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/README.md>.
  Execution follows closure of the Qwen/shared regression gate.
- The first FLUX attempt exposed another shared control-plane defect before
  generation: Model Manager's batch resource planning blocked HTTP health and
  library requests. Both single/batch planning handlers evaluated snapshots,
  model inventory, and candidates synchronously in the HTTP event loop.
  These operations and response serialization now run through `asyncio.to_thread`;
  graph admission still revalidates authority, and no model defaults were changed.
  A new regression fails both old handlers by demonstrating that a scheduled
  heartbeat cannot run during planning. The fixed endpoints pass 147 focused
  tests and the complete backend suite (2,461 passed, 54 skips, 6,894 subtests).
  The idle server was restarted; live Model Manager now permits library use
  and returns a health response in 9 ms during planning. Evidence is under
  `flux/flux2-klein-text-to-image/model-manager-control-plane.json`.
- The final backend lint audit found a misplaced `BlockInstanceV2.routeSelection`
  type annotation after an unrelated function's return and an unused local in
  composition rebuilding. The optional field is now declared on the correct
  TypedDict, with a type-introspection regression; the existing composition
  validation call is retained without its unused assignment. Focused route and
  composition tests pass (13); the final full suite is being rerun.
- FLUX then passed parameter editing, expand/collapse export parity and
  save/refresh, but real task `8rcGEaRVf9Cw` failed before loading weights:
  the loader's separate mostly-Qwen workflow allowlist rejected Klein's
  catalog-declared `text2image` selection. Explicit loader selections now
  validate against the same pinned upstream workflow snapshot as the catalog,
  and retain the complete selected component bundle. All 94 exact workflow
  selections pass the new regression; unknown families/routes still fail,
  mandatory Qwen/MiniMax selection and unscoped legacy behavior are preserved.
  This changes loader scoping, not execution/publication admission.
  Research: Diffusers Modular quickstart `get_workflow()`/`init_pipeline()` and
  the installed pinned `ModularPipeline` constructor's `workflow` argument.
  Focused loader/identity tests: 77 passed, 12 skipped, 200 subtests. Restart and
  real FLUX rerun follow; no image was produced by the failed task.

Final delivery checkpoint:

- FLUX.2 Klein text-to-image passed the real frontend lifecycle and completed
  task `772RuGGIrJ01`: insert, parameter locality, advanced guidance check,
  nested containment, save/refresh, collapsed/expanded export parity, Run,
  preview/media retrieval, and unchanged creator defaults. The full test took
  approximately 2.7 minutes. There were no browser page errors.
- Review candidate: `flux/flux2-klein-text-to-image/museum-workshop.webp` in the
  new evidence root. It is 1024×1024, four steps, guidance 1.0, seed 20260905,
  BF16/model CPU offload and no quantization. Exact prompts, model pin, workflow,
  receipt and screenshots are beside it. Visual inspection finds coherent
  workshop composition/materials; the requested label is not clearly rendered.
  No showcase/publication approval has been inferred.
- Full backend: 2,464 passed, 54 skipped, 6,988 subtests. Full client `check`,
  shared controls (2), mocked Studio browser suite (126), deployed Qwen smoke,
  Qwen real generation/reuse/structural edits, all-94 subtree audit, and the
  FLUX live demo pass. The backend has been restarted with the loader and
  planning fixes, and the checked client bundle is deployed.
- Remaining shared performance issue: workflow saves during concurrent full
  Model Manager planning took approximately 9–25 seconds; one diagnostic queue
  request also exceeded three seconds. Offloading planning removed its direct
  event-loop block, but does not prove all control-plane latency is resolved.
  Profile workflow validation/serialization and planner contention next, before
  broadening the FLUX campaign. Do not hide this by reducing model defaults.
  Budget 1–3 focused hours for this performance follow-up, then continue the
  remaining five cached FLUX generation routes and FLUX structural-edit proofs
  within the 6–12-hour FLUX tranche estimate above.

The new Qwen outputs are 256-square, two-step **technical checks**, not proposed
showcase assets. Existing high-quality assets remain in their earlier campaign
folder; they are not copied into this new review batch.

### Continuation — save responsiveness, then remaining cached FLUX

- Reproduced blocking storage/serialization in all five workflow route paths
  with HTTP-heartbeat regressions. Workflow operations now run off-loop under
  the existing shared persistence lock; summary listings retain bounded metadata
  instead of rereading all graph snapshots on every request.
- The clean-browser baseline save took 368 ms; the earlier 9–25-second delay is
  intermittent, not a claim that every save took that long. Profiling its actual
  13-form Model Manager request found 5.96 seconds in inventory scanning and
  4.36 seconds in 13 repeated optional-runtime catalog inspections (10.87 seconds
  total with profiler overhead).
- Inventory was decoding large tokenizer/weight-map payloads to find loader
  class names. These payloads are now excluded; normal/custom configuration JSON
  remains eligible. The measured scan dropped from 5.879 seconds to 0.082 seconds,
  with an identical class-index digest across all 97 cached repositories.
- Batch planning now shares one lazy runtime inspection per request, retaining
  fresh checks on the next request and at graph admission. Official Python
  `asyncio.to_thread`, Transformers configuration, Diffusers configuration, and
  installed pinned loader config constants were reviewed before these changes.
- Next acceptance: full backend regression, cold-restart frontend save/refresh
  under planning, then real frontend runs for cached FLUX Klein edit, Klein Base
  text/edit, and Kontext text/edit. Record asset quality separately from lifecycle
  correctness; keep creator defaults unchanged and use explicit instance values.
- Save follow-up completed: full backend regression **2,470 passed, 54 skipped,
  6,993 subtests**; workflow heartbeat, cache invalidation/bounding and cancelled
  save-notification regressions pass. The backend was restarted. The final live
  browser save under concurrent Model Manager planning took **191 ms**; both
  batch plans finished in **1.33–1.36 seconds**. Receipts are in
  `performance/after/`. The profiled 13-form request now takes 1.04 seconds
  instead of 10.87 seconds on this machine; these are observed measurements,
  not a universal latency guarantee.
- FLUX.2 Klein edit passed the complete real frontend lifecycle as task
  `frSbt8RXlxQ2`: source image entered through its file field, complex edit
  instruction, instance-local dimensions/steps/seed/guidance, nested containment,
  collapsed/expanded parity, Save, refresh, Run and persisted media. The 1024-square
  result changed the blue velvet to burgundy while preserving the main workshop
  composition. Four steps, guidance 1.0, BF16/model CPU offload, no quantization;
  creator defaults unchanged. Evidence: `flux/flux2-klein-edit-image/`.
  The earlier attempts failed in the test harness before submitting a run:
  its Advanced disclosure had not been opened, and the float control normalizes
  `1` to `1.0`. The shared input label correctly inherits its shell's ID; no
  input-component change was necessary or retained.
- FLUX.2 Klein Base text-to-image passed the same frontend lifecycle as task
  `Rk1Z_RY6nYhO`, with 208 seconds of measured backend execution (4.5 minutes
  for the browser test). Exact creator recommendation: 1024-square, 50 steps,
  guidance 4.0, BF16/model CPU offload, no quantization. Evidence:
  `flux/flux2-klein-base-text-to-image/`. The scene is coherent, but the requested
  label still contains a lettering error; retain this limitation for review.
- Kontext's exact cached revision uses the FLUX.1 [dev] Non-Commercial License
  v1.1.1. The license was retrieved through `hf_hub_download`, not a direct-file
  downloader, and read before scheduling that family. Requested explicit
  confirmation of permitted non-production evaluation or separate commercial
  authorization for `24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d` before execution.
  This does not block the Apache-licensed Klein/Base routes.
- Klein Base edit passed the frontend lifecycle as task `qzJ0-Y3jJ4Fc`:
  447 seconds backend execution, 8.5 minutes complete browser lifecycle,
  1024-square/50 steps/guidance 4.0/BF16/model CPU offload/no quantization.
  Blue-to-burgundy velvet editing preserves the main composition, with minor
  fine-detail changes. Four of six cached FLUX routes now have current generation
  evidence; this is not completion of FLUX structural-edit/User Node qualification.
- Deployed frontend port 8088 also passed the save-under-planning test:
  Save 307 ms, planning 1.419 seconds, edited prompt retained after refresh.
- Post-change Studio regression reached 124/126 passing. Both failures traced
  to required Hub dataset fixtures returning HTTP 429, not model execution.
  Added an opt-in, SHA-256-verified local runtime-input fixture route, populated
  using `hf_hub_download`; production download/admission behavior is unchanged.
  The Wan fixture test now passes. Full regression rerun follows with frozen
  frontend sources and verified fixture cache; do not count it as green yet.
- That full rerun is now green: **2 shared-control + 126 Studio browser tests**,
  zero failures (9.0 minutes Studio), plus `npm run check` including unit/build/
  bundle and exact registered-catalog checks. Shared controls and interface
  mutations are mocked client proofs, not substitutes for live generation.
- Generalized the existing Qwen structural-execution browser test for cached
  FLUX Klein. Replacement is performed by a real pointer drag; link deletion
  now waits for Arrange geometry and selects a pointer-reachable segment instead
  of assuming its midpoint is exposed. The complete generation rerun is active.
- FLUX Klein structural rerun passed (1.7 minutes) as task `QCdN2VXiBCr2`:
  compatible internal prompt replacement, outgoing-edge deletion, pointer
  reconnection, Save/refresh, identical instance values, containment and real
  1024-square generation with a complex observatory prompt. The remaining FLUX
  reusable User Node lifecycle is not covered by this workflow-only test.
  The image and readable parameters are in `flux/flux-klein-v2-structural-execution/`.
  No product UI change was needed for the test harness's edge-geometry or
  JSON-escaped quote assertions. The sign lettering is imperfect; no automatic
  showcase approval is inferred.

- [x] Qwen catalog insertion/ownership and progressive-expansion browser failures resolved.
- [x] Complete the remaining adoption/reconnection focused browser test.
- [x] Current Qwen editing/reuse/generation proofs complete (separate attributed lifecycle tests).
- [x] New Arrange/insertion fixes deployed and clean-browser smoke passed.
- [x] Required client/backend/browser checks complete with results recorded.
- [x] First FLUX.2 Klein frontend demonstration and asset ready.
- [x] Workflow-save/batch-planning latency follow-up (9–25 seconds observed).
- [ ] All six cached FLUX routes qualified through the frontend.

## Goal

Finish the Qwen Image vertical slice with no known visual, editing,
persistence, or execution defects, then apply the same source-neutral contract
and automated conformance gates to all 94 pinned Modular Diffusers workflows.

The result must provide both experiences without maintaining two graph types:

- a registered Cluster is the immutable, collapsed, one-node way to run a
  reviewed workflow;
- expanding that instance reveals the exact active Modular Diffusers hierarchy
  as ordinary Blocks with ordinary controls, ports, selection, movement,
  resizing, connection, and subtree-save behavior;
- changing the instance is copy-on-write. The registered catalog definition is
  never changed, while the workflow retains the exact modified instance;
- a changed root or any changed subtree can be saved as a User Node with its
  current prompts, parameters, model/component bindings, topology, interfaces,
  layout, and immutable provenance;
- browser refresh and backend restart restore the same instance, not current
  catalog defaults;
- execution continues through MoDiff's one backend graph executor and the
  pinned Diffusers `init_pipeline()` composition path.

This plan covers the 94 public workflows discovered from the pinned Modular
Diffusers snapshot. The library endpoint currently also contains one standard
Diffusers composite (`WanTI2VPipeline`), so the UI reports 95 Diffusers Cluster
entries. That extra route receives the shared canvas/catalog checks, but it is
not counted as one of the 94 Modular workflows.

## Upstream contract used by this plan

The implementation follows the current official Hugging Face Diffusers
Modular documentation and the exact pinned source revision, not inferred
behavior from display names:

- `ModularPipelineBlocks` are reusable units that can be composed and can
  contain sub-blocks;
- `SequentialPipelineBlocks` pass intermediate state from one block to the
  next;
- `LoopSequentialPipelineBlocks` own loop-member blocks and loop state;
- `AutoPipelineBlocks` select a branch from supplied trigger inputs;
- `get_workflow()` selects the exact active workflow;
- any valid block/subtree may become a pipeline through `init_pipeline()`;
- adding, removing, and swapping blocks changes the effective pipeline inputs,
  outputs, and components and must be rebuilt and validated, not visually
  simulated by the client.

The upstream API is experimental. All imported structure, inputs, outputs,
components, class identities, and revisions therefore remain snapshot-pinned
and hash-covered.

## Qwen issue ledger and catalog-wide invariants

Every issue below occurred, or was exposed by manual Qwen testing, during the
current implementation. Each row defines the wider regression that must run
against all 94 definitions so a family-specific patch cannot hide the same
class of defect elsewhere.

| Area | Qwen issue observed | Required invariant for all 94 |
| --- | --- | --- |
| Definition identity | Generated `BlockDefinitionV2` identity became stale after catalog/compiler changes. | A reproducible build regenerates every definition; client, backend, served catalog, embedded instance, and content hash agree byte-for-byte before insertion. |
| Old workflow loading | Stale execution `bindings` prevented Studio loading. | Invalid execution metadata is reported and recoverable without preventing an empty graph from opening; no unvalidated binding reaches execution. |
| Insertion | Click/drop appeared to do nothing while definition materialization was slow. | Every click/drop shows immediate per-entry progress, inserts once, and reports a bounded actionable failure. |
| Empty graph drop | A Cluster could fail with malformed presentation state or a stale receipt. | All structurally admitted Clusters and Blocks insert into an empty graph with valid V2 presentation and no pre-existing task node. |
| Renderer parity | Registered Cluster, changed instance, and User Node used different renderers, actions, controls, ports, and field shapes. | One Block V2 frame and projection renders every root/container/subtree; registration changes provenance and permissions only. |
| Expansion containment | Children appeared outside the root; collapsed user dimensions constrained expanded content. | Expanded bounds are derived bottom-up from visible descendants and ignore compact collapsed dimensions. Every visible descendant lies within its immediate visible parent. |
| Nested containment | Content overflowed intermediate containers. | The same recursive bound rule applies at every depth, including after child resize, move, collapse, re-expand, save, and refresh. |
| Initial disclosure | Expanding the root also expanded every nested container. | A fresh root expansion reveals immediate children; deeper containers begin collapsed and expand independently. |
| Layout | Nodes overlapped, scattered, or inherited stale saved coordinates. | Deterministic sibling layout resolves collisions per parent while preserving intentional user positions; legacy positions normalize once without changing execution. |
| Resizing | Root/internal Block resizing clipped children, hid content, or lost links. | Compact size persists separately from derived expanded bounds. Resize handles, connector trays, and links remain usable at the minimum size. |
| Ports | Some root and internal Blocks had no visible inputs/outputs. | Every visible subtree derives typed boundary ports from public bindings and semantic edges crossing its boundary. Leaf ports are unchanged. |
| Collapsed links | Links disappeared when an internal container was shrunk/collapsed. | Crossing edges project to the nearest visible boundary port and always resolve back to the exact semantic leaf endpoint. |
| Inert alternatives | Qwen VAE/control-VAE auto alternatives appeared as disconnected active nodes in text-to-image. | Unselected upstream auto/conditional alternatives remain in the immutable source snapshot but never render or execute as active workflow nodes. Customized/user-owned or newly connected nodes are never hidden by this rule. |
| Nested required inputs | Qwen Edit supplied the source image to representative text/VAE owners but not to an earlier active nested `resize` leaf, so the exact graph failed only at runtime. | Every external media/data edge fans out to every selected active leaf declaring that socket; every required runtime input socket has an incoming edge. Inactive alternatives receive neither edges nor execution. |
| Stale inactive branches after refresh | Older finalized graphs could preserve unmatched exact execution roles as infrastructure, leaving useful-but-inactive VAE/ControlNet branches visible and disconnected in text-to-image. | Re-finalization preserves only actual loader/preview/export infrastructure. Skipped Conditional/Auto placements remain available in the left library for compatible routes but do not survive as active graph nodes. |
| Loader visibility | It was unclear which model was used; the model selector was disabled. | Every executable Cluster exposes its reviewed loader/components and exact repository/revision at the appropriate expanded level. Compatible same-family choices are explicit and preserve shared fields. |
| Creator defaults | Prompts were blank or technical test overrides looked like defaults. | Creator/model-card starter values are visibly labeled and immutable catalog defaults remain unchanged. Qualification overrides live only in the test instance/receipt. |
| Parameter-only edit | A small parameter edit changed presentation/interface and made the node look unrelated. | Editing one value changes only that instance value and copy-on-write ownership state; graph, interface, layout, other values, and source definition remain byte-identical. |
| Structural edit | Move/add/replace/delete/reconnect behavior was incomplete or silently changed semantics. | Valid structural edits update the effective graph and exact composition recipe; invalid edits remain visible, identify the implicated node/edge, and offer a precise Fix action. |
| Conversion semantics | A changed registered instance looked like a different canvas type. | Copy-on-write does not change the renderer or effective interface. Only the provenance/ownership badge and allowed save choices change. |
| Save choices | Update existing, save as new, and workflow-only behavior was unclear. | Registered instance: save new or keep workflow-only. User-owned instance: update existing, save new, or keep workflow-only. Other open instances remain unchanged. |
| Persistence | Reloading previously reset model/prompts/parameters. | Save, refresh, clean browser, and backend restart preserve definition snapshot, values, model selection, ports, topology, expansion, sizes, and positions. |
| Multi-instance isolation | Automatic model/route state could couple separate nodes. | Two instances and two workflows have independent values, effective graphs, routes, and presentation. |
| Run authority | Expert/non-Auto runs were blocked by warnings such as projected memory risk. | Expert mode blocks only true missing/invalid graph, runtime, artifact, or required-input errors. Risks remain prominent warnings with explicit run-anyway/recovery actions. |
| Qualification language | `Prepare qualification run` appeared in the ordinary node and obscured normal Run. | Qualification is a developer/evidence action, never the primary user workflow. Normal runnable instances use the application Run/Queue controls. |
| Worker failure | AMD SVM/OOM killed a worker, leaving stale progress and a misleading backend-disconnected state. | Worker death atomically fails the run with preserved stderr classification, model/resource context, and actionable retry/unload/lower-memory guidance. Backend health and worker health are distinct. |
| Long loop progress | Exact nested denoise execution remained at the outer graph percentage while upstream `progress_bar()` advanced only in worker stderr, making a healthy Qwen run look stuck. | Every exact reviewed loop owner bridges the upstream Modular Diffusers progress-bar boundary into queue/activity telemetry with step count, elapsed time, ETA, heartbeat, and clean step-boundary cancellation; the pinned block tree is not modified. |
| Fixed-sequence workflow identity | Qwen Edit Plus/Layered use a reviewed internal `default` identity, but their upstream `SequentialPipelineBlocks` do not publish `_workflow_map`; forwarding `workflow="default"` made Diffusers reject model loading. | Keep the reviewed identity in the graph/runtime contract, but forward `workflow` to Diffusers only for a real named upstream workflow. Fixed block sequences initialize directly, as required by the upstream API. |
| Fixed-sequence nested runtime paths | The selected catalog indexed `text_encoder.resize` as one dotted compatibility segment while the fixed upstream block tree exposes the real path `text_encoder` → `resize`; execution therefore loaded every model component and then rejected the first nested leaf. | Named workflows execute and validate their `get_workflow()` path. Internal `default` definitions execute and validate the full immutable unpruned path. Every fixed-definition executable parameter path must equal its hierarchical placement path; dotted display/index keys never cross the backend boundary. |
| Nested decoded media | Qwen Image Layered correctly returned decoded PIL images grouped as `batch → layers`, but the shared Preview node recognized only a flat PIL list, misclassified the nested collection as latents, requested an unrelated VAE, and let the graph finish without visible media. | Preserve the upstream nested value inside the Modular graph, then normalize decoded nested PIL batches only at the generic Preview/output boundary. A run is not complete evidence until its typed media reaches Studio and is persisted. |
| Shared logical input fan-out | Qwen Image Layered exposed `layers` on its loop owner and on earlier/later nested preparation leaves. Editing the root control changed only the loop owner, so preparation retained the upstream default of four layers even when the workflow requested two. | One admitted top-level ModularPipeline argument compiles to one public V2 control whose binding and canonical mirrors cover every selected active leaf consuming that argument. Existing Studio/export consumers and newly discovered hierarchical consumers are merged under one authority; inactive, output-only, shadowed-optional, and differently owned fields are excluded. |
| Default mutation | Low-resource debugging could silently alter default parameters. | Resource presets and test overrides are explicit reversible instance changes. No debug path mutates registered defaults. |
| Discovery | 94 Clusters and more than 1,000 contextual block rows produced an unusable flat list. | The catalog uses nested facets and one visible entry per semantic identity. Contexts are metadata/choices, not duplicate rows. |
| Status labels | Catalog-only, structurally available, installed, runnable, and promoted states were conflated. | Separate badges describe structure, artifact installation, runtime/resource qualification, and publication; none imply another. |

## Definition of done and evidence levels

No Cluster is marked complete from a screenshot or a compiler test alone.
Evidence is cumulative:

1. **Schema** — immutable source, definitions, hashes, ports, hierarchy, and
   dependencies validate without model import or download.
2. **Canvas contract** — insertion, progressive expansion, containment,
   resize, ports, links, controls, parameter edit, structural edit, undo, save,
   refresh, restart, and instance isolation pass.
3. **Execution contract** — a visible-frontend run reaches the pinned official
   Diffusers path, emits progress, handles cancellation/failure, and produces
   typed media.
4. **Output qualification** — the current definition/revision/resource recipe
   produces a reviewable asset with its full receipt.
5. **Publication** — a human approves that exact asset/receipt before public,
   `liveProof`, showcase, or Auto-eligible flags are enabled.

The all-94 guarantee means no known issue and passing automated invariants for
all 94 definitions. It does not pretend that static tests can prove hardware
execution for model snapshots that are not installed.

## Work order

### Gate 0 — Inventory and reproducibility

1. Freeze the pinned Diffusers revision, the 94-workflow manifest, unpruned
   hierarchy snapshot, generated V2 catalog, and backend compiler revision.
2. Add one manifest that records every workflow's family, modality, task,
   hierarchy depth, active definition identity, artifact repositories,
   installation state, execution/resource status, and evidence paths.
3. Fail both client and backend gates on stale generated identities, duplicate
   workflow identities, duplicate semantic block identities, invalid edges,
   missing endpoints, or mutable artifact revisions.
4. Keep audit scratch data in temporary ignored directories and delete it after
   each gate.

### Gate 1 — Qwen Image release gate

No work is promoted to another model family until this gate passes.

1. Run the full 10-route Qwen structural/browser matrix from a clean empty
   workflow using the production-served frontend.
2. For Qwen text-to-image, perform and persist:
   - creator-quality complex prompt and negative prompt;
   - width, height, steps, guidance, seed, offload, and compatible model choice;
   - root and nested expand/collapse and resize;
   - parameter-only edit with zero collateral graph/interface/layout changes;
   - add, replace, remove, move-out, and reconnect of compatible Blocks;
   - invalid connection followed by the highlighted Fix recovery;
   - root, intermediate container, loop, and leaf save-as-User-Node;
   - workflow-only, save-new, and update-existing persistence decisions;
   - save, refresh, backend restart, and multi-instance isolation.
3. Run the modified workflow from the visible frontend in collapsed and
   expanded states. Compare execution graph fingerprints and deterministic
   receipt fields; preserve both generated assets.
4. Repeat route-specific required-input and persistence checks for Qwen
   image-to-image, inpaint, three ControlNet routes, Edit, Edit inpaint, Edit
   Plus, and Layered.
5. Do not qualify an inactive auto branch as a missing/disconnected node. Do
   fail any active or user-connected orphan.

### Gate 2 — Generic all-94 conformance harness

Implement a table-driven suite over the pinned manifest, not 94 model-specific
frontend branches. For every Modular workflow it must verify:

- current generated identity and exact revision;
- deterministic `BlockDefinitionV2` compilation;
- unique node/edge/parameter/port identities;
- valid edge endpoints and type-compatible crossing bindings;
- selected upstream auto/conditional workflow only;
- exact recursive parent chain and maximum declared depth;
- progressive initial collapse state;
- collapsed and fully expanded execution graph equality;
- boundary ports and links at every visible container depth;
- bottom-up containment after minimum/maximum resize and child move;
- parameter-only copy-on-write locality;
- structural mutation validation and recoverable invalid state;
- save/normalize/export/import/refresh byte equivalence;
- two-instance isolation;
- no hidden Hub request, runtime install, or remote-code enablement during
  browsing, insertion, expansion, or planning.

Run browser coverage exhaustively for insertion/collapse/persistence where it
is bounded, and use risk-selected live browser cases for each modality, block
kind, maximum hierarchy depth, optional branch shape, loop shape, and component
loading pattern. A real model run remains per-route evidence and is never
faked by this structural suite.

### Gate 3 — Nested catalog and duplicate removal

Replace the four flat Hugging Face lists with a searchable tree:

- **Diffusers Cluster Nodes**
  - modality: Image, Video, Audio, Multimodal;
  - task: Text to Image, Image to Image, Inpaint, Text to Video, and so on;
  - family: Qwen Image, FLUX, Stable Diffusion XL, Wan, and so on;
  - workflow/model variant.
- **Modular Diffusers Block Nodes**
  - family;
  - role/kind: Load Components, Encode/Condition, Prepare Inputs/Latents,
    Denoise, Decode/Postprocess, Preview, Auto/Conditional, Loop, Other;
  - one exact semantic definition row.
- **Diffusers Component Nodes**
  - component role/type;
  - family compatibility and reuse contexts.
- **User Nodes**
  - workflow/source family;
  - user versions.

Duplicate policy:

- a Cluster identity is `(provider, pipeline class, workflow id, immutable
  revision)`;
- a Modular Block identity is its exact pinned block-definition identity/hash;
- one block used in multiple placements is shown once, with all compatible
  family/workflow/path contexts in its details and insertion context chooser;
- aliases and friendly labels improve search but do not create another row;
- context-specific variants remain separate only when their contract, child
  tree, components, or exact composition recipe differs;
- search matches every hidden breadcrumb/context and expands only matching
  branches.

All rows keep immediate insertion feedback and are draggable both onto a
top-level graph and into a compatible expanded Block. Incompatible drops are
kept visible with a node-targeted validation/Fix explanation.

### Gate 4 — Generic same-family model and route behavior

Generic means contract-compatible, not arbitrary cross-task switching.

1. Declare model choices in backend execution/admission data with exact
   repository, revision, component schema, parameter aliases, and supported
   workflows.
2. Offer choices in a shared loader control only when the selected model is
   compatible with the Cluster's task and effective public interface.
3. On model/route change, diff the reviewed definitions by stable semantic
   source identity:
   - retain shared nodes, current compatible values, user positions, public
     ports, and connections;
   - add/remove only route-specific subtrees;
   - preserve disconnected user additions and explain any newly incompatible
     connection;
   - never reset an unchanged prompt/parameter merely because a loader changed.
4. Rebuild valid changes through the exact pinned `init_pipeline()` recipe.
5. Store route drafts independently so switching away and back restores each
   route's prior instance state.

### Gate 5 — Cached-model asset campaign

Asset generation starts only after the corresponding definition passes Gates
1 and 2. It is performed through the visible frontend so the normal user flow,
queue, progress, media preview, save/refresh, and error handling are exercised.

Rules:

- model installation uses the application's Hugging Face Hub snapshot pull;
  never `wget`, mutable URLs, or an unrecorded manual weight copy;
- use exact immutable revisions and `trust_remote_code=false` unless a separate
  reviewed authorization explicitly permits a bounded remote-code route;
- start with already complete cached repositories to avoid unnecessary
  downloads and disk churn;
- use official model-card/creator recommendations as the starting values;
  quality overrides are stored in the workflow and receipt, not registered
  defaults;
- preserve one dated, isolated review directory containing only this campaign;
  include asset, thumbnail/contact sheet where relevant, frontend screenshot,
  workflow export, definition hash, model/artifact revisions, prompt, negative
  prompt, all effective parameters, resource recipe, timing, and run logs;
- do not mix historical assets into the current review folder;
- after human approval, retain the approved evidence and preview cache deletion
  per model family before rotating to uncached models.

Current cache candidates to reconcile against exact route revisions first
include Qwen Image 2512/Edit/Edit 2511/Layered and Qwen ControlNet, FLUX/FLUX
Kontext/FLUX.2 Klein variants, Stable Diffusion XL and its reviewed adapters,
Wan 2.1/2.2 and Wan Animate variants, Z-Image, ERNIE Image, and MiniMax Music 3.
Cache presence alone does not prove that every required component or exact
revision is complete, licensed, resource-qualified, or compatible with the
current pinned Diffusers runtime.

Prioritized output order after Qwen:

1. one high-quality image path per installed family;
2. image edit/inpaint/control paths using curated source media;
3. audio with creator-recommended duration/steps;
4. video with motion-specific prompts and native recommended dimensions/frame
   counts, rejecting static, pixelated, or temporally glitchy results;
5. additional routes that reuse the resident family before cache turnover.

### Gate 6 — Regression, evidence, and promotion

1. Run the complete client unit/type/lint/build/bundle gate.
2. Run the complete backend suite and cross-runtime definition/hash fixtures.
3. Run the complete mocked Studio browser suite.
4. Restart the backend and run a clean-browser production-bundle smoke.
5. Reconcile Git so generated media, caches, temporary workspaces, tokens,
   logs, browser reports, and local configuration remain ignored and absent.
6. Stage definition-specific promotion receipts only for human-approved output;
   approval of technical behavior does not imply showcase or Auto approval.

## Immediate implementation slices

The first implementation pass is intentionally bounded and testable:

1. add the all-94 conformance audit for identity, hierarchy, selected branches,
   boundary connectivity, collapsed/expanded execution equality, and
   parameter-only locality;
2. change the Modular Block catalog from contextual-placement rows to unique
   semantic definitions with retained searchable contexts;
3. add nested catalog breadcrumbs/groups and duplicate-count regression tests;
4. run the complete Qwen 10-route production-browser lifecycle;
5. run a real modified Qwen text-to-image frontend generation into a new dated
   review folder;
6. run full gates before moving to the first cached non-Qwen family.

## Blockers and required acknowledgements

- **Disk/resource:** current free storage must be rechecked before each large
  pull and video/audio run. Cache rotation must not delete a model until all
  reusable routes for that family have approved evidence or an explicit
  decision says otherwise.
- **Gated/license-controlled artifacts:** any repository requiring license
  acceptance or usage/revenue/territory acknowledgement needs the exact model
  revision and explicit operator acknowledgement recorded before installation
  or execution. Existing acknowledgements do not transfer to a new revision.
- **Remote repository Python:** arbitrary `block.py` execution remains outside
  this phase. Declarative imports remain remote-code-disabled; a sandbox and
  separate explicit authorization are required for reviewed remote Python.
- **Hardware:** AMD/ROCm compatibility and memory qualification are per recipe.
  A structurally valid Cluster may remain execution-blocked until its exact
  recipe passes. Expert mode still reports all issues and only hard-blocks
  requirements that make execution impossible or unsafe to submit.
- **Output approval:** public promotion requires a human decision on the exact
  current asset and receipt. Old or technically valid but poor-quality assets
  are not reused as showcase approval.

## Time estimate

Assuming no new upstream incompatibility:

- Qwen final structural/edit/persistence gate plus one real modified
  text-to-image proof: 0.5–1.5 focused days, with model runtime included;
- generic all-94 conformance harness and defect fixes it exposes: 1.5–3 days;
- nested catalog, semantic deduplication, and browser coverage: 1.5–3 days;
- cached-model execution campaign: approximately 3–7 days of elapsed machine
  time, heavily dependent on video/audio runtimes and reruns for quality;
- full regression/evidence cleanup: 0.5–1 day;
- uncached/gated model rotation: additional time per download, license gate,
  and hardware qualification, estimated only after the cache-to-route manifest
  is complete.

The functional all-94 structural/catalog target is therefore approximately
4–7 focused engineering days. Real high-quality execution evidence for every
route cannot be honestly bounded to that window because several snapshots are
not installed or legally/resource qualified; cached routes will be completed
first and reported separately.

## Progress checklist

- [x] Official Modular Diffusers behavior and pinned source boundary reviewed.
- [x] Historical Qwen defect classes converted into catalog-wide invariants.
- [x] Cache-to-route/evidence manifest generated. Current machine-readable
      inventory:
      `data/review/diffusers-94-hardening-2026-09-05/diffusers-94-qualification-manifest.json`.
- [x] Qwen 10-route clean live-backend browser contract rerun. Current receipt:
      `data/review/diffusers-94-hardening-2026-09-05/qwen-v2-family-lifecycle/frontend-result.json`.
- [x] Real modified Qwen text-to-image proof generated. The technical
      structural-edit receipt is under
      `data/review/diffusers-94-hardening-2026-09-05/qwen-v2-structural-execution/`;
      the current quality asset and receipt are isolated under
      `data/review/diffusers-94-assets-2026-09-05/qwen-image-2512-block-v2-live/`.
- [x] All-94 generic structural conformance audit green: 94 workflows, 1,794
      semantic nodes, maximum compiled placement depth 5, no duplicate
      workflow identities, boundary interfaces present, recursive containment,
      progressive nested disclosure, collapsed/expanded execution equality,
      parameter-only locality, zero disconnected active execution leaves, and
      zero unbound required runtime inputs.
      Containers are projections rather than executable leaves; inactive
      conditional alternatives are excluded from the selected graph. Evidence:
      `data/review/diffusers-94-hardening-2026-09-05/all-94-structural-audit.json`.
- [x] Nested Hugging Face catalog and semantic Modular Block deduplication
      implemented: 1,051 placement contexts now render as 598 exact semantic
      Block rows while preserving all contexts for compatible insertion.
- [x] Deployed Qwen production nested-Block smoke passes from an explicitly
      empty graph: skipped text-to-image VAE/ControlNet alternatives are absent,
      every visible nested container exposes connectors, resize preserves its
      incident links, progressive expansion retains recursive containment, and
      sibling projections do not overlap. Screenshot:
      `data/review/diffusers-94-hardening-2026-09-05/qwen-production-nested-smoke.png`.
- [x] Sequential duplicate-media-writer regression fixed generically. Public
      direct-media outputs now follow the final selected writer/postprocessor
      unless an exact reviewed placement mapping deliberately pins another
      writer. The affected 9 Qwen and 2 Anima definitions were regenerated and
      repinned; the compiler suite also proves that any duplicate writers have
      one terminal writer and that the public output binds to it.
- [x] Remaining Qwen admission execution wave complete. Qwen image-to-image,
      inpaint, and all three ControlNet routes now pass insert, edited values,
      expand/collapse parity, save, refresh, real visible-frontend execution,
      and persisted media output. The Edit-family wave exposed and now has a
      generic fix for nested required-input projection: external values fan out
      to every selected active leaf declaring that socket, and the 90-route
      compiler gate rejects any required runtime input without an incoming
      edge. Qwen Edit now passes that real frontend rerun as task
      `iaMyam0yQPwU`, and Edit Inpaint passes as task `pD_0nDpId1ho`. Edit Plus
      also exposed two fixed-sequence API mismatches which are now covered by
      backend and all-route compiler tests: `workflow="default"` is no longer
      forwarded to a class without `_workflow_map`, and nested executable paths
      use the full immutable tree rather than dotted catalog shorthand. The
      corrected Edit Plus single- and multi-reference frontend reruns pass.
      Layered now persists its decoded nested PIL collection correctly; its
      first visible proof then exposed a generic shared-control fan-out defect.
      The compiler now merges existing reviewed consumers with every selected
      hierarchical leaf consuming the same ModularPipeline argument, and all
      90 registered BlockDefinitionV2 routes compile with zero ambiguity. The
      corrected Layered visible-frontend rerun passed as task
      `qxjE5KXAasA9`: the saved `layers = 2` value reached the loop owner and
      all three selected nested consumers, exactly two images were persisted,
      save/refresh remained byte-identical, and collapsed/expanded execution
      stayed equivalent. Together with the earlier text-to-image proof, all 11
      Qwen admissions now have current real-frontend execution evidence.
- [ ] Cached image-family asset batch ready for review.
- [ ] Cached audio/video batch ready for review.
- [ ] Full release regression green.
- [ ] Approved definition-specific promotion receipts staged.
