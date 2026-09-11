# Demoable Hugging Face Nodes Plan — 2026-08-27

> **Status correction — 2026-09-01:** The separate Cluster renderer and
> automatic Cluster-to-User-Node reconstruction used by this demo are legacy
> behavior. They are no longer accepted as completion because reconstruction
> can change controls, public ports, actions, and connections. Registered
> Clusters and User Nodes must use one composite-node system with copy-on-write
> workflow customization. The authoritative contract, migration plan, current
> checklist, and estimates are in the
> [Unified Composite Node Contract and Implementation Plan](unified-composite-node-implementation-plan-2026-09-01.md).
> Checked legacy rows below retain historical test evidence only.

## Outcome required

By the 2026-08-27 demo, MoDiff should visibly prove:

1. One Diffusers Cluster each for image, video, and standalone audio can be
   inserted, edited, saved, refreshed without parameter reset, and executed.
2. A first-party Cluster and a User Node have the same composite canvas surface.
   A structural edit updates that workflow instance copy-on-write, preserves
   its controls, ports, values, dimensions, preview, and external connections,
   and leaves the catalog definition unchanged.
3. A new library node or an existing top-level canvas node can be placed inside
   an expanded composite. The wrapper identity and explicit public interface
   stay stable, and the topology survives collapse, Save, browser refresh,
   re-expansion, and execution.
4. User Nodes and Cluster Nodes are never nested.
5. A changed User Node offers three explicit persistence choices:
   **Update existing User Node**, **Save as new User Node**, or
   **Keep changes only in this workflow**.
6. A remote-code-disabled Hub import can install an exact commit, preview its
   declarative contract, summarize the import, save it to User Nodes, survive
   refresh, and execute when the contract is supported.

## Product decisions

### Cluster customization is copy-on-write, not conversion

Parameter edits are always instance-local and do not change first-party
ownership. Viewing or changing a parameter therefore keeps the node a Cluster.
Moving an existing owned child is presentation-only. The first actual topology-
changing action—adopting a node, deleting or reconnecting an internal node, or
requesting direct structure editing—must:

1. Copy the exact effective graph inside the same workflow instance.
2. Apply the user's structural action as one undoable mutation.
3. Preserve wrapper identity, repository, immutable revision, parameters,
   explicit controls and ports, external edges, position, size, and preview.
4. Remove incompatible first-party qualification and Auto authority without
   changing rendering or source provenance.
5. Keep the change workflow-only until the user explicitly saves or updates a
   reusable User Node definition.

The registered definition stays immutable. `Cluster` remains its library and
provenance category; it is not a different renderer. No destructive conversion
or automatically persisted fork is part of the accepted flow.

### No nesting

- A Cluster or User Node may not become a child of another Cluster/User Node.
- Dropping a Cluster/User Node over an expanded composite leaves it top-level
  and displays a clear notice.
- Ordinary nodes may be adopted into one expanded composite at a time.
- Moving a child between composites detaches it from the old instance and
  adopts it into the new instance in one undoable action.

### Contextual names

A workflow-only customized registered block keeps its existing instance name.
It is not renamed merely because its topology changed.

Saving a new reusable definition defaults to:

`<current block name> — <workflow title>`

Names remain editable. Identity is an opaque ID, so two definitions with the
same display name do not overwrite one another.

### User Node persistence choices

- **Update existing User Node** saves the current topology and exposed
  parameters under the current User Node ID. The current workflow instance is
  already updated; future insertions use the updated reusable definition.
- **Save as new User Node** generates a new ID, uses the contextual default
  name, saves the exact current topology, and points the current workflow
  instance at the new definition.
- **Keep changes only in this workflow** updates only the embedded workflow
  snapshot. The reusable library definition remains unchanged.

No choice should silently rewrite other open workflow instances. Existing
instances retain their embedded snapshot until the user deliberately chooses
to refresh them from the reusable definition.

## Priority and completion status

Completed rows report their measured verification result for the original demo
critical path. The active H5 expansion below still has license-bound live
qualification and publication-review work; the bounded H6 speech expansion is
live-qualified. The arbitrary-Python sandbox remains an explicitly separate
security project.

| Priority | Work                                                                                                                      | Functional implementation |                                                                                            Verification | Completion target                         |
| -------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------: | ------------------------------------------------------------------------------------------------------: | ----------------------------------------- |
| P0       | MiniMax Music 3 visible-frontend install and standalone-audio generation                                                  |                  Complete | Real frontend install/edit/save/refresh/parity/generate test passed in 2.0 min with the cached snapshot | Complete                                  |
| P0       | Replace legacy automatic conversion with shared V2 composite instances and copy-on-write customization                    |               In progress |                              Exact two-instance Qwen regression plus V1-to-V2 migration required | 1.5–2 days for Qwen demo; 4–6 days full |
| P0       | Drag an existing top-level ordinary node into an expanded User Node; move between User Nodes; reject Cluster/User nesting |                  Complete |                                               Real pointer drag plus mocked persistence/choice coverage | Complete                                  |
| P0       | Update existing / Save as new / Keep only in workflow choices with contextual naming                                      |                  Complete |                                                       Mocked browser Save/refresh/isolation test passed | Complete                                  |
| P1       | Complete declarative Hub import wizard with remote code disabled                                                          |                  Complete |                                                            Mocked test and real official-Hub E2E passed | Complete                                  |
| P2       | Execute arbitrary repository `block.py` in a separately authorized sandbox                                                |          2–4 working days |                   Security boundary, escape, network/filesystem, cancellation, and resource-abuse tests | Not safe to promise for the next-day demo |

The tomorrow-demo critical path is P0. The declarative Hub importer is included
as P1 because it does not execute repository Python. Arbitrary Python remains
explicitly unsupported and fail-closed until the P2 sandbox is qualified.

## Implementation sequence

### Phase 1 — real audio proof

- [x] Exact MiniMax pipeline/block contract and five-node execution graph.
- [x] Revision-bound license acknowledgement UI and authorization received.
- [x] Install the exact 23-file snapshot through the visible Model Manager.
- [x] Insert the Cluster, edit prompt/lyrics/steps/duration, Save, refresh, and
      compare parameters.
- [x] Compare collapsed and expanded API graphs.
- [x] Generate real audio, copy the asset and receipt into the review folder,
      and inspect the media metadata.

### Phase 2 — legacy structural conversion evidence (superseded)

These checked rows prove what the 2026-08-27 build did. They do not satisfy the
2026-09-01 shared-composite contract and must be replaced by the active V2 plan.

- [x] Extract Cluster-to-User-Node conversion into one reusable application
      action used by the Cluster header, library drop, and canvas drag paths.
- [x] Rename the visible action to **Customize as User Node**.
- [x] Detect a normal library-node drop within an expanded Cluster, convert the
      instance, expand the resulting User Node, and apply the drop.
- [x] Detect an existing ordinary canvas node dropped within an expanded User
      Node and adopt it with correct relative coordinates.
- [x] If dropped within an expanded Cluster, convert the Cluster first and then
      adopt the node.
- [x] Support moving a child from one User Node to another.
- [x] Reject Cluster/User Node nesting consistently for library and canvas drag.
- [x] Make each adoption/conversion one undoable history transaction.

### Phase 3 — explicit reusable-definition choices

- [x] Materialize the current User Node definition from either expanded
      children or the collapsed embedded snapshot.
- [x] Add **Update existing**, **Save as new**, and **Keep in workflow** actions.
- [x] Generate workflow-context names using the active workflow title.
- [x] Preserve current workflow values, exposed ports, origin provenance, and
      immutable artifact revisions for all three choices.
- [x] Ensure Update does not mutate already-open instances silently.
- [x] Ensure Save as new produces a new opaque ID and a separately insertable
      library row.

### Phase 4 — declarative Hub import

- [x] Turn the current repo/commit form into a bounded wizard state machine.
- [x] Wait for exact-revision installation and show progress/errors.
- [x] Read and validate `mellon_pipeline_config.json` without executing repo
      code.
- [x] Preview block hierarchy, components, ports, artifact commit, and rejected
      fields.
- [x] Show admission and remote-code summary before confirmation.
- [x] Save the validated definition to User Nodes and preserve it across Save
      and browser refresh.
- [x] Run one executable remote-code-disabled real-Hub E2E against
      `diffusers/FLUX.2-klein-4B-modular@62ac375aa5308588f111fcd12115f5c54a8b1f4f`
      with every component pinned to an exact immutable revision.
- [x] Keep repositories requiring `block.py` blocked with a precise explanation
      until the separately authorized sandbox exists.

## Test order

Functional tests precede browser E2E so failures remain fast and attributable.

1. Pure graph tests: adoption coordinates, move between containers, no nesting,
   copy-on-write graph edits, stable explicit boundaries, definition
   identity/version choices.
2. Store tests: undo/redo, save/update/new behavior, instance isolation, JSON
   round trip.
3. Mocked browser tests: library drop, existing-child movement, copy-on-write
   structural customization, three persistence choices,
   Save/refresh/re-expand.
4. Real frontend tests: Qwen registered-block customization and workflow-owned
   block execution,
   MiniMax standalone audio, and safe declarative Hub import.
5. Final smoke: image, video, audio demo receipts and assets collected under a
   single review index.

## Historical demo evidence — 2026-08-26

- MiniMax Music 3: exact 23-file/28.52 GB revision installed through the visible
  Model Manager after the revision-bound license acknowledgement. Multiline
  prompt/lyrics, duration, steps, seed, dtype, device, and offload settings were
  equal before and after Save/browser refresh. Collapsed and expanded exports
  were equal, and task `KqYefJpE0m8c` produced a 44.1 kHz stereo WAV.
- Qwen Image: the visible frontend test edited and persisted Cluster prompt,
  width, height, steps, seed, device, dtype, and offload parameters; generated
  task `fetVV_JvHMy4`; dragged an existing Data Viewer across the expanded
  Cluster boundary with real pointer input; automatically converted only that
  instance to `Qwen Image — Text To Image — Workflow 1`; connected the decoder
  image branch to the adopted node; preserved the modified topology through
  collapse, Save, refresh, and re-expansion; and generated task
  `urYf-nTyEoe_` from the customized User Node. Both runs produced byte-equal
  256px technical images, while the second run also executed the retained Data
  Viewer branch.
- Safe Hub import: the official FLUX.2 Klein Modular repository and both of its
  component repositories were installed at exact commits, inspected without
  importing repository Python, previewed, imported with
  `trust_remote_code=false`, saved, refreshed byte-for-byte, and executed as
  task `f9r--ix0rxHr`. It produced a 1024px image with media hash
  `sha256:bytes:2d1f79356886ebfe9d89692939be7fd5093b78c55a0f00387546c87402f002f8`.
- Release verification: the complete backend gate passes with 2,185 passed,
  51 skipped, one warning, and 6,557 subtests; the complete client
  check/unit/typecheck/lint/build/bundle gate passes; all 114 mocked Studio
  browser tests pass; and the focused optional-runtime cutover browser
  regression passes.

## Remaining post-demo work

The model-execution and asset portions of the initial demo are implemented and
evidenced. The composite presentation, customization, and persistence portion
was reopened by the 2026-09-01 architecture correction and is complete only
when the [unified V2 plan](unified-composite-node-implementation-plan-2026-09-01.md)
passes. The separate arbitrary-repository-Python
sandbox remains fail-closed until it can enforce filesystem/network
restrictions, time and memory limits, cancellation, explicit operator
authorization, and escape tests. This does not block declarative Mellon import
or first-party reviewed block execution. Showcase-quality prompt
and parameter tuning is also intentionally separate from the technical
qualification assets.

### Ordered remaining Diffusers priority — 2026-08-27

Work must proceed in this order:

1. Complete shared release engineering before adding more definitions:
   extend incompatible resident-model cleanup and resource admission from Auto
   runs to Expert/Cluster runs, run the full regression, then generate
   definition-specific promotion receipts and enable public flags only after
   the corresponding output is approved.
2. Complete the strengthened visible-frontend lifecycle for the 14 already
   admitted routes. The nine remaining SDXL combinations, LTX-2
   image-to-video, and both LTX 0.9.8 routes are now green; the active remainder
   is Wan 2.2 text-to-video and image-to-video.
3. Only after steps 1 and 2 are green, promote the remaining 45 contract-only
   workflows family by family. Each batch must retain immutable artifacts,
   official Diffusers/Modular Diffusers behavior, exact conditionals and state
   flow, license/access gates, bounded resource planning, and a real visible-
   frontend lifecycle receipt before publication.

Structural catalog coverage is already 94/94. This ordering concerns runtime
admission, live qualification, and publication; it must not relabel a
contract-only definition as executable from schema inspection alone.

Shared resource handling is now complete. The pre-run cache boundary applies
to both Auto and Expert/Cluster graphs, compares model family, immutable
artifact, loader topology, dtype, quantization and offload recipe, preserves an
identical reusable resident pipeline, and releases incompatible app-owned node,
model, Modular component, disk-offload and allocator state before loading the
next family. A visible-frontend regression warmed both FLUX.2 Klein Base routes
and left 15.1 GiB allocated / 15.7 GiB reserved, then ran LTX-2 in the same
worker without a restart. Task `n3gcydVUYz6O` records the exact
`Flux2KleinBaseModularPipeline` -> `LTX2ModularPipeline` cleanup, 14 released
cached nodes, successful allocator trimming, no cleanup errors, and a completed
H.264/AAC output. Evidence is under
`test-review/frontend-pipeline-e2e-2026-08-26/cross-model-memory/`.

The same idle-only resource boundary now also protects destructive Model
Manager cache turnover. Before deleting an immutable Hub revision, the backend
revalidates the hash-bound plan, releases app-owned node/model/Modular
Diffusers/offload references, collects and trims the allocator, and only then
removes snapshot bytes. The response contains a typed `runtimeRelease` receipt.
This closes a live Linux failure where a 44.36 GB LTX cache directory was
deleted while roughly 30 GB of its safetensors blobs remained open in the
worker. Focused backend coverage passes with `88 passed, 40 subtests`; client
action tests, lint, and typecheck are green. The first repaired visible
frontend turnover evicted the already-qualified LTX-2 snapshot, reclaimed
92,074,139,514 bytes immediately, preserved workflows/definitions, and left no
deleted Python file descriptors. The frontend qualification harness records
the release receipt on subsequent turnovers.

## Active release-hardening tranche — 2026-08-27

Work is ordered by user-visible correctness first and broad regression cost
last. A checked item means both implementation and its focused regression are
complete; live-backend and full-suite rows remain distinct proof levels.

### H1 — guided semantic composition

- [x] Inspect every expanded User Node for missing sockets, invalid directions,
      incompatible types, invalid container crossings, and incompatible sealed
      Modular workflow-state identities.
- [x] Discover compatible insertion points for an adopted ordinary node and
      offer an explicit one-click edge splice instead of guessing silently.
- [x] Make insertion/replacement/deletion one undoable graph mutation and retain
      the resulting topology through collapse, Save, refresh, and re-expansion.
- [x] Preserve the exact upstream pipeline/workflow/block identity in User Node
      provenance, while clearly distinguishing ordinary MoDiff graph edits from
      reviewed upstream `ModularPipelineBlocks` composition.
- [x] For reviewed upstream block-tree edits, reconstruct the selected workflow,
      apply only exact pinned block operations, and call upstream
      `init_pipeline()` so Diffusers recomputes inputs, outputs, components, and
      configs. Unknown classes, paths, revisions, state, or container kinds must
      fail closed.

Focused proof: the graph/store suite covers typed splice discovery, invalid
direction/type/state rejection, one-step undo, provenance sealing, and
replacement/deletion. The live backend additionally rebuilt the pinned Qwen
workflow after a bounded structural operation and returned an
`init_pipeline()` receipt without loading model weights.

### H2 — legacy live lifecycle and isolation evidence

- [x] Retain a compatible inserted node in Qwen's actual execution graph, then
      reconnect, Save, refresh, Run, replace, delete, and verify persistence.
- [x] Run **Update existing**, **Save as new**, and **Keep only in workflow**
      against the live backend; prove a second open instance is unchanged.
- [x] Add a visible two-Cluster/two-workflow isolation proof covering model,
      prompt, and parameter edits before and after refresh.
- [x] Prove undo/redo of automatic Cluster-to-User conversion and node adoption.
- [x] Prove backend User Node save failure and interrupted workflow save leave
      the original Cluster/User Node recoverable and do not leak changes to
      another workflow.

The legacy Qwen test supplies the retained real model branch and both
generation receipts. The generic live-backend topology test separately proved
replacement, deletion, invalid-socket diagnostics, interrupted workflow save,
and recovery without paying the Qwen model-loading cost for each mutation.

### H3 — executable declarative Hub import

- [x] Add an immutable, remote-code-disabled declarative integration fixture
      whose sidecar resolves entirely to admitted MoDiff/Diffusers nodes.
- [x] Prove import -> contract preview -> User Node -> Save -> refresh -> Run
      against the real backend. Keep repositories requiring `block.py` Python
      preview-only until the separate sandbox is authorized and qualified.

### H4 — release regression and publication boundary

- [x] Run the complete client check/build/bundle gate, full backend suite, and
      complete mocked Studio browser suite after H1-H3.
- [x] Restart the backend and run a clean-browser smoke, including legacy
      workflow/User Node migration fixtures.
- [x] Generate definition-specific promotion receipts for the two explicitly
      approved technical outputs: Qwen Image 2512 text-to-image and MiniMax
      Music 3 text-to-audio. Receipt lookup is admission-specific, so the Qwen
      approval cannot promote the five other workflows sharing its execution
      profile. Both exact admissions now publish `liveProof: true` while
      `executable` and `autoEligible` remain runtime/resource-gated. Unreviewed
      routes still require their own output approval and receipt.

Current regression checkpoint: the post-FLUX.2-Klein-Base backend gate is
green with 2,198 passed, 53 intentional skips, 1 warning, and all 6,579
subtests passed. The complete client unit/typecheck/build gate is green, and
the post-change mocked Studio browser suite passed all 114 tests in 6.2
minutes. The verified client build was mirrored into backend `web/` while
preserving backend-owned content; a clean backend restart returned healthy
same-origin API responses, served the byte-identical built shell, established
a visible websocket connection, completed workspace startup, and produced no
browser page errors. Its screenshot is
`test-review/frontend-pipeline-e2e-2026-08-26/production-static-smoke.png`.
No active core graph/lifecycle technical gate remains. Future publication
receipts remain intentionally blocked on explicit approval of each associated
technical output; arbitrary repository Python remains a separate sandboxed
security project rather than a prerequisite for declarative Hub import.

### H5 — Diffusers definition qualification expansion

- [x] Audit the current 94-definition Diffusers catalog. Fifty execution
      admissions across 49 definitions are graph-qualified and insertable; the
      other 45 definitions remain structurally insertable but contract-only.
- [x] Register the three admitted FLUX-family Modular pipeline identities in
      the closed client model/profile registry so authoritative backend
      execution specifications are retained instead of silently discarded.
- [x] Preserve the exact saved Modular pipeline identity when canonical legacy
      workflows are hydrated and adopted; do not fall back to a standard
      pipeline merely because both identities share one repository.
- [x] Make Cluster preparation recover from a missing/stale capability snapshot
      by refreshing authoritative runtime data and retrying the exact
      specification identity and content hash.
- [x] Bind every image-oriented Modular Diffusers `ModelsLoader` revision to
      the immutable reviewed artifact commit. This closes the revision gap in
      39 execution specifications; all 50 admitted routes now expose the same
      exact artifact revision in their sealed loader binding.
- [x] Admit the pinned FLUX.2 Klein checkpoint's repository-scoped
      `Qwen2TokenizerFast` serialization as the exact Transformers 5
      `Qwen2Tokenizer` compatibility alias. Arbitrary tokenizer substitutions
      remain rejected.
- [x] Promote the official non-distilled **FLUX.2 Klein Base 4B** workflows
      from contract-only discovery to native Modular graph admission without
      aliasing them to the four-step distilled class. The text and
      image-conditioned definitions retain `Flux2KleinBaseAutoBlocks`, 50-step
      defaults, classifier-free guidance 4, and exact artifact revision
      `a3b4f4849157f664bdbc776fd7453c2783562f4d`.
- [x] Review and pin the public Apache-2.0 component snapshot: 21 selected
      files, 15,964,212,614 weight bytes, no repository Python, and no duplicate
      root single-file checkpoint. The checked-in review records every selected
      weight's Hub LFS SHA-256 and keeps Auto, Gallery, and live proof closed.
- [x] Finish the visible frontend FLUX.2 Klein Base qualification. The browser
      activated the exact optional runtime, installed the revision-bound
      21-file snapshot through Model Manager, inserted both definitions,
      changed their parameters, saved and refreshed their workflows, expanded
      all 10 text and 13 edit placements, proved collapsed/expanded graph
      equality, and generated real WebP outputs. The task ids are
      `cqSH2AL_5Z31` and `xvcKj8ikhGyw`; their media hashes are
      `sha256:bytes:8898a0947624f2b57b7dc8fc766be65841fda3fc6877fc190c268ff814095729`
      and
      `sha256:bytes:eaa30ce003fca5e2205ae6e80b5a506539e6b0c8ab41d47a854a6b77b9bf6a4b`.
      Evidence is preserved under
      `test-review/frontend-pipeline-e2e-2026-08-26/flux2-klein-base-clusters/`.
      The outputs remain qualification-only and the public flags are unchanged
      pending an improved showcase run and explicit approval.
- [x] Generate separate showcase candidates through the same visible frontend
      lifecycle at 768x768 and 24 steps. The text-to-image task
      `WHgJeEi0XV7q` produced
      `sha256:bytes:34e8b314a0cde65a0e0536d01f77076e9aa2d103cce1850c91eea5c256da3fc4`;
      the image-edit task `6oSITqhuz31a` produced
      `sha256:bytes:f3bc89e15235b111bddd079496ab4e7097b06b59f1c4e6fcc5815a85f124e3f1`.
      Both nondefault prompts and sampling parameters survive Save/browser
      refresh exactly. The candidates and receipt are under
      `test-review/frontend-pipeline-e2e-2026-08-26/flux2-klein-base-showcase/`.
      They remain `showcaseApproved: false`; no promotion receipt or public flag
      may be issued until explicit review approval.
- [x] Qualify **Flux2 Klein — Text To Image** and **Flux2 Klein — Edit Image**
      through the visible frontend: catalog insertion, parameter changes,
      workflow Save, browser refresh, exact parameter comparison, block-tree
      expansion, collapsed/expanded API graph comparison, pinned-revision model
      load, and real generation all pass in one browser test.
- [ ] Promote either definition's immutable public `executable`, `autoEligible`,
      or `liveProof` flag only after its generated output and definition-specific
      receipt receive explicit approval.
- [x] Qualify **Flux — Text To Image**, **Flux — Image To Image**,
      **Flux Kontext — Text To Image**, and **Flux Kontext — Edit Image**
      through the visible frontend. Each route now proves insertion,
      nondefault parameter edits, workflow Save, browser refresh with exact
      parameter equality, complete reviewed block-tree expansion,
      collapsed/expanded graph equivalence, exact-revision loading, and real
      generation.
- [x] Preserve the official FLUX split-block geometry contract: image-to-image
      preprocessing receives the saved requested size, while the decoder
      receives the actual post-denoise width and height. This covers Kontext's
      documented normalization to a supported resolution without rewriting the
      saved Cluster parameters.
- [x] Admit only the two pinned FLUX.1 repositories' Transformers 5
      `T5TokenizerFast` serialization as the exact `T5Tokenizer` compatibility
      alias. Unrelated repositories and tokenizer substitutions remain
      rejected.
- [x] Select Z-Image as the next family: its exact reviewed artifact is already
      installed, Apache-2.0 licensed, supports both official `text2image` and
      `image2image` Modular workflows, and does not require auxiliary model
      repositories.
- [x] Preserve the official Z-Image image-encoder geometry contract by exposing
      `height` and `width` on the split VAE encoder and binding the saved
      Cluster dimensions into that node.
- [x] Qualify **ZImage — Text To Image** and **ZImage — Image To Image** through
      the visible frontend with nondefault edits, Save/browser refresh exact
      equality, complete block expansion, collapsed/expanded graph equality,
      exact-revision execution, and real generated outputs.
- [x] Select the installed Apache-2.0 Qwen Image Edit family ahead of SDXL and
      qualify **Qwen Image Edit — Edit Image** through the same complete visible
      frontend lifecycle across its exact 17-block `image_conditioned`
      hierarchy.
- [x] Qualify **Qwen Image — Image To Image** across its distinct 16-block
      `image2image` workflow with saved width, height, strength, prompt, source,
      steps, seed, and runtime overrides.
- [x] Qualify **Qwen Image — Inpaint** across its 18-block `inpainting`
      workflow, including exact source and mask paths before and after browser
      refresh.
- [x] Qualify **Qwen Image Edit — Inpaint** across its exact 20-block
      `image_conditioned_inpainting` workflow. The qualification fixed two
      execution-contract defects found by the real run: the source image is now
      connected to prompt conditioning, and the split graph no longer forces
      saved 256px dimensions onto the model's official image-conditioned latent
      geometry.
- [x] Qualify both admissions of **Qwen Image Edit Plus — Default** through the
      visible frontend: explicitly select `edit_image` or
      `multi_image_reference_edit`, preserve the selected admission and exact
      one/two-image input list across Save and browser refresh, expand the exact
      17-block hierarchy, prove collapsed/expanded graph equality, and execute
      the pinned revision.
- [x] Qualify **Qwen Image Layered — Layer Decomposition** with nondefault
      `layers`, `resolution`, and `max_sequence_length`. Preserve those values
      across Save/refresh, prove the exact 18-block collapsed/expanded graph,
      and carry the upstream nested layer batch through the app as a three-item
      `image_collection` without dropping outputs.
- [x] Qualify the complete official Qwen Image ControlNet workflow set:
      **Control Image**, **Controlnet Image2Image**, and **Controlnet
      Inpainting**. Bind and execute—not only persist—`max_sequence_length`,
      `controlnet_conditioning_scale`, `control_guidance_start`, and
      `control_guidance_end`; preserve the exact control/source/mask paths and
      all nondefault values through Save and browser refresh; and prove the
      respective 16-, 20-, and 22-block expanded graphs are API-equivalent to
      their collapsed Clusters.
- [x] Preserve a prepared Cluster's run authority when backend dynamic-field
      refreshes echo an identical sealed value after expansion. Exact no-op
      echoes now bypass history and authority invalidation; any differing
      sealed value remains rejected.
- [x] Qualify the next installed, admitted Diffusers family: **Wan — Text To
      Video** now covers the Apache-2.0 Wan 2.1 1.3B native Modular pipeline.
      The qualification added the missing execution bindings for width,
      height, and `max_sequence_length`; retained the exact live backend
      execution spec in the client registry; and admitted only the pinned
      1.3B checkpoint's official `T5TokenizerFast` serialization as the exact
      Transformers 5 `AutoTokenizer` compatibility alias.
- [x] Prove the Wan video route through the visible frontend: catalog
      insertion, nondefault prompt/seed/guidance/size/frame/fps/sequence edits,
      workflow Save, browser refresh with exact equality, complete reviewed
      block expansion, collapsed/expanded API graph equality, immutable
      revision loading, real generation, and MP4 persistence all pass.
- [x] Run the full backend, client, and mocked-browser regression gates after
      the Wan qualification changes: backend `2185 passed, 51 skipped, 6557
      subtests`; the complete client `npm run check` gate passed; and all 114
      mocked Studio browser tests passed.
- [x] Qualify **Wan Image2 Video — Flf2V** through the visible frontend. Bind
      `max_sequence_length` to the official Modular text-encoding block; retain
      distinct first- and last-frame paths, prompt, geometry, generation
      controls, and immutable revision across Save/browser refresh; expand all
      17 reviewed placements; prove collapsed/expanded graph equality; execute
      the native Modular route; and persist a valid nine-frame MP4.
- [x] Qualify the installed Wan 2.1 single-image-to-video workflow through the
      visible frontend. The exact 36-file model closure is recognized by Model
      Manager; the workflow retains its prompt, source image, geometry,
      sequence length, frame count, FPS, seed, dtype, and offload controls
      across Save/browser refresh; all 15 reviewed placements expand; the
      collapsed and expanded API graphs are identical; and the native Modular
      route persists a valid nine-frame MP4. Continue with another installed
      Apache-2.0 Diffusers family. Use SDXL only after its revision-bound
      OpenRAIL++ acknowledgement is explicitly available in the visible flow.
- [x] Qualify **Ernie Image — Text To Image** as the next Apache-2.0 family.
      Install the exact 24-file, 31.65 GB closure through visible Model
      Manager; preserve the exact Modular definition, pinned revision, prompt,
      1024px geometry, step count, seed, guidance, dtype, and offload values
      across Save/browser refresh; expand all nine reviewed placements; prove
      collapsed/expanded API graph equality; execute the reviewed equivalent
      standard `ErnieImagePipeline`; and persist the generated WebP. Public
      execution flags remain unchanged pending explicit output approval.
- [x] Preserve the original upstream conditional trees in a separate reviewed
      no-weight snapshot before promoting LTX-2 as a native Modular route. The
      companion contract covers all 34 pipeline classes and 94 workflows with
      632 static block definitions, 1,051 unpruned placements, and 73
      conditionals. It records trigger order, every branch/default, complete
      presence/absence truth tables (including rejected combinations), and the
      selected branch plus active-leaf trace for every advertised workflow
      predicate. A detached read-only
      `GET /huggingface/modular-conditionals` endpoint publishes it without
      changing the existing resolved definition identities. Focused contract
      and API tests pass. A pinned no-weight upstream test also proves that
      LTX-2 with `num_frames` omitted includes `LTX2DurationStep` and a
      `duration_head` component, while supplying `num_frames` removes both and
      rebuilds the resulting interface through `init_pipeline()`.
- [x] Consume the reviewed conditional companion in the node-library client,
      materializer, and expansion UI. The active workflow instance must show
      its selected branch while retaining inactive official alternatives. For
      LTX-2, prove both `num_frames` supplied (duration prediction skipped) and
      `num_frames` omitted (duration branch active) against upstream
      `get_execution_blocks(...)`/`init_pipeline()` before claiming native
      execution.
      Implement this with the upstream presence semantics, not model aliases:
      `ConditionalPipelineBlocks` selects only from trigger-input presence,
      `AutoPipelineBlocks` selects the first present trigger (or its declared
      default), and `SequentialPipelineBlocks.get_execution_blocks()` adds
      earlier intermediate-output names to the active input set before later
      selectors run. The official API contract is
      <https://huggingface.co/docs/diffusers/api/modular_diffusers/pipeline_blocks>.
      Any edited block tree must be rebuilt with `init_pipeline()`; changing a
      previously created pipeline's copied `blocks` property is not a runtime
      mutation. The official lifecycle contract is
      <https://huggingface.co/docs/diffusers/modular_diffusers/modular_pipeline>.
      The reviewed companion and client projection therefore carry each
      placement's exact upstream `intermediate_outputs`; final public outputs
      are not incorrectly treated as later selector inputs.
      The visible LTX-2 frontend proof now passes without loading model weights:
      a newly inserted Cluster treats Studio-seeded execution values as
      omitted, displays the active `duration` selector and branch, then treats
      an instance edit to `numFrames=9` as explicitly present and displays the
      selector as skipped plus the official duration block as an inactive
      alternative. The explicit-presence receipt and value survive workflow
      Save/browser refresh; clearing the value restores the active duration
      branch. Collapsed and expanded graph exports remain identical. Evidence
      is preserved under
      `test-review/frontend-pipeline-e2e-2026-08-26/ltx2-conditional-cluster-ui/`.
- [x] Add a revision-bound LTX-2 Community License acknowledgement before any
      native LTX-2 qualification. The exact reviewed revision is
      `47da56e2ad66ce4125a9922b4a8826bf407f9d0a`; its published terms include an
      annual-revenue threshold, redistribution duties, and use restrictions.
      The visible install and run paths now share one revision-bound policy
      key; focused unit and visible mocked-browser tests prove cancel versus
      explicit-confirm behavior. User/company acknowledgement has not been
      inferred from another model's terms. On 2026-08-27, the user explicitly
      acknowledged this exact revision and confirmed that the planned use is
      below the stated annual-revenue threshold or separately authorized. The
      fresh native qualification completed through the visible Model Manager
      and its revision-bound terms dialog. To make the app's 64 GiB cache-volume
      safety reserve fit, the already-qualified
      `black-forest-labs/FLUX.1-Canny-dev` local cache was safely evicted
      through Model Manager; its saved workflows and canonical definitions
      remain available and the weights are recoverable by exact redownload.
      The exact 45-file, 92,074,139,514-byte snapshot installed successfully.
      The visible Cluster test then passed insert, expand an official internal
      block, edit prompt and execution parameters, Save, browser refresh,
      visible value comparison, collapsed/expanded graph equality, and native
      video-with-audio generation. Task `Vz4mWmZ3gAii` completed in 118.4 seconds
      on the clean ROCm worker and produced an H.264/AAC MP4 with SHA-256
      `8d6c532f5f275c58def492305db4084806904ccf66b4532fc7c4e1ddea1ea965`.
      The run also fixed two large-Cluster UI races: offscreen node focus now
      temporarily suspends React Flow virtualization until `fitView()` finishes,
      and Diffusers Cluster insertion awaits the reviewed Modular Conditional
      companion so its hierarchy cannot rematerialize during an edit. A first
      retry on a long-lived worker confirmed that cached weights from another
      large model can cause a global OOM; the exact same reviewed parameters pass
      after clearing those cross-model allocations with a clean worker. Evidence
      is under `test-review/frontend-pipeline-e2e-2026-08-26/ltx2-native-licensed/`.
      Keep both equivalent-standard LTX-2 admissions and any future native
      admissions unpublished until the fresh generated output is reviewed.
- [x] Complete the strengthened lifecycle for all nine previously outstanding
      SDXL admissions: ControlNet Union image-to-image/inpainting; IP-Adapter
      image-to-image/inpainting; IP-Adapter + ControlNet text/image/inpainting;
      and IP-Adapter + ControlNet Union text/image. Every route now passes
      visible insertion, nondefault parameter edits, workflow Save, browser
      refresh with exact equality, official conditional hierarchy expansion,
      collapsed/expanded API-graph equality, exact-revision preparation, and
      real generation. The browser qualification found and fixed one shared
      schema defect: a connected IP-Adapter guider could transiently inject an
      undefined local `guidance_scale` during expansion. Connected guiders now
      suppress that local callback while retaining the reviewed upstream CFG
      component. PNG/WebP assets are preserved under
      `test-review/frontend-pipeline-e2e-2026-08-27/sdxl-remaining-clusters/`.
      The evidence writer now keeps one JSON receipt per route rather than
      overwriting the previous selection; the consolidated receipt rerun is
      still required before promotion. All outputs remain qualification-only
      pending output approval.
- [x] Complete the strengthened LTX-2 image-to-video lifecycle. Conditioned
      generation exposed one exact upstream runtime prerequisite: the official
      pipeline applies H.264 CRF recompression and imports PyAV. MoDiff now
      declares pinned PyAV 18.1.0 in its managed composite media-codec runtime
      and requires it only for LTX-2 image/reference modes. The client now also
      expands exact reviewed `satisfiesProfiles` aliases, so the active
      Transformers + OpenCV + PyAV composite safely satisfies the pinned
      Transformers-main requirement without weakening digest checks. The real
      frontend test passed insert, internal prompt edit, nondefault dimensions,
      steps, frames and FPS, Save, browser refresh with exact equality,
      collapsed/expanded graph parity, and real video-with-audio generation in
      6.6 minutes. Task `tKvuz8NMqo4i` produced a 166,124-byte MP4 with media
      hash
      `sha256:bytes:909049257dbd79b177f8f760eeb48bf55211fe56f2832c38875609563d13e945`.
      Evidence is under
      `test-review/frontend-pipeline-e2e-2026-08-27/ltx2-native-licensed/`.
- [x] Complete the strengthened **LTX — Text To Video** lifecycle. Hugging Face
      Hub 1.28 correctly rejected the former partial local snapshot because the
      official `LTXConditionPipeline` addresses 12 nested
      `vae/text_encoder`/`vae/transformer` aliases in addition to the root
      components. MoDiff's exact reviewed selection now includes all 34 paths;
      Model Manager repaired them without duplicate byte transfer because the
      aliases resolve to the same immutable blobs. Task `5cpBN6QNPIan`
      completed after Save/refresh and collapsed/expanded parity, producing
      media hash
      `sha256:bytes:200460890584e7abacc82e5052379503b0be09e937349288d4686e3fe18209bf`.
- [x] Complete the strengthened **LTX — Image To Video** lifecycle. Expanding
      finalized dynamic execution fields exposed a real reset race: mounting a
      reviewed field replayed its initial action and could change the sealed
      mode from `image_to_video` to `text_to_video`. Materialized execution
      fields now carry `suppressInitialFieldAction`; signal relay remains live,
      but already-finalized on-change/on-signal actions are not replayed on
      render. Task `_R2l1dv0tmme` passed exact persistence and graph parity and
      produced media hash
      `sha256:bytes:2331aacca6e3a589a455b5f82f1f1629f6093073dc6b07145d6db54517fed7ca`.
      Both LTX receipts and MP4s are under
      `test-review/frontend-pipeline-e2e-2026-08-27/remaining-admitted-video-clusters/`.
- [ ] Complete the two remaining admitted routes: **Wan 2.2 — Text To Video**
      and **Wan 2.2 Image2 Video — Image To Video**. The strengthened harness
      installs each exact snapshot through Model Manager and records one
      route-specific persistence, parity, runtime-cleanup, generation and
      media-provenance receipt. The exact 43-path, 126,199,294,813-byte Wan 2.2
      text-to-video transfer is active through the visible frontend at revision
      `5be7df9619b54f4e2667b2755bc6a756675b5cd7`. Model Manager first rejected it
      with only 157 GiB free because the snapshot plus the 64 GiB safety reserve
      did not fit; after the repaired frontend eviction reclaimed the completed
      LTX-2 local copy, admission passed with 243 GiB free.

### H6 — prioritized Transformers speech-to-text lifecycle

- [x] Keep this phase bounded to the requested speech families. The first
      reviewed pair is **Whisper Tiny — Speech to Text** and **Whisper Tiny —
      Speech Translation** at the exact Apache-2.0 revision
      `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`; broader text, vision, and
      multimodal Transformers Cluster expansion remains deferred.
- [x] Implement the four-node reusable speech graph (load audio, load exact
      processor/model, transcribe or translate, and preview structured text)
      and bind task, language, timestamp mode, chunk length, stride, dtype,
      device, and immutable revision into the expanded User/Cluster execution
      nodes. This follows the official Transformers Whisper task-token contract:
      `transcribe` retains the source language and `translate` produces English.
      See <https://huggingface.co/docs/transformers/main/en/model_doc/whisper>.
- [x] Prove both modes generate structured text from the pinned 5.12-second
      LibriSpeech fixture through the visible frontend. The technical outputs
      already identify the spoken sentence and retain segment timestamps.
- [x] Rerun the strengthened lifecycle proof after the Diffusers tranche:
      change nondefault audio/task/language/timestamp/chunk/stride/runtime
      values, Save, refresh, compare the exact embedded instance, expand all
      four nodes, compare collapsed/expanded exports, and execute both
      transcription and translation again. The visible frontend proof passed
      against the real backend with transcription task `9J5O3M2Ik1Mj` and
      translation task `Qb3IDT81WXgi`; both decoded the pinned 5.12-second
      fixture with segment timestamps. Evidence is preserved under
      `test-review/frontend-pipeline-e2e-2026-08-26/transformers-asr/`.
- [x] Add one additional reviewed CTC speech family only after the Whisper
      lifecycle remains green in the final release regression. Do not broaden
      unrelated Transformers families in this phase. **Wav2Vec2 Base 960h —
      Speech to Text** is now bound to exact revision
      `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`. Its visible frontend proof
      repaired and activated the optional Transformers runtime through the app,
      cached the exact repository, preserved the instance through Save/browser
      refresh, expanded all four nodes, proved collapsed/expanded parity, and
      executed task `A23zwjxbPlqo` with word timestamps. The CTC output and
      screenshots are preserved in the same Transformers ASR evidence folder.

The FLUX.2 qualification tasks are `06_fORfy8Bw2` (text-to-image) and
`VTL8AsHT5m8b` (edit image). Both used
`black-forest-labs/FLUX.2-klein-4B@e7b7dc27f91deacad38e78976d1f2b499d76a294`,
256 x 256 output, two inference steps, bfloat16, and model CPU offload. The
first generated WebP has SHA-256
`407f58958b2f6d23556f08f3088db866e106252c6145b6270643d872cd5fdcac`;
the edit generated WebP has SHA-256
`d569d38cb4431bc4642c090f95fb86fa4b3557be9a461eeb1ff61f24ad96cff8`.
The combined receipt and review assets are under
`test-review/frontend-pipeline-e2e-2026-08-26/flux2-klein-clusters/`.
They remain technical qualification outputs with `showcaseApproved: false` and
do not change any publication flag.

The FLUX.1 qualification tasks are `juZGQKFp89Km` (FLUX text-to-image),
`JsLeZkX47tAH` (FLUX image-to-image), `vQ5bUUPMtCLh` (Kontext text-to-image),
and `xvaI4_KbTbVr` (Kontext edit image). They used the immutable revisions
`black-forest-labs/FLUX.1-dev@3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`
and
`black-forest-labs/FLUX.1-Kontext-dev@24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d`,
two inference steps, bfloat16, and model CPU offload. The generated WebP
SHA-256 values are `878feaedfebe7f83bc53b6d867d03c03efff7d4e40dc21a646ab8086dc35569c`,
`9889d2963805309c0e7db9aa1f2a7eae180ea70fe700a59f46796651f02bbf36`,
`d68c11ac395fcf861c1d0fc4508601bd47d6783bdf117fd8580a6e16c644107a`,
and `0a4fd280fc4e7f772021ad04c62ae54617031ca36ef0d988b3d4c7ce10df3261`
in the same order. The combined receipt and review assets are under
`test-review/frontend-pipeline-e2e-2026-08-26/flux1-clusters/`. These are
technical qualification outputs with `showcaseApproved: false`; all four
definitions' public promotion flags remain unchanged pending explicit output
approval.

The Z-Image qualification tasks are `qBbM662AL1dj` (text-to-image) and
`bdkPTwcDGQjR` (image-to-image). Both used
`Tongyi-MAI/Z-Image-Turbo@f332072aa78be7aecdf3ee76d5c247082da564a6`,
256 x 256 output, two inference steps, bfloat16, and model CPU offload. The
generated WebP SHA-256 values are
`12decbe27650d19aca2dfbf819b484f079865dbe71629c43ea0dcd10a1dd0b68`
and `8a15b4d31bc89e5003f5b019f1c091f37487157b733a63e556ad6f928d10498b`.
The combined receipt and review assets are under
`test-review/frontend-pipeline-e2e-2026-08-26/z-image-clusters/`. These remain
technical qualification outputs with `showcaseApproved: false`; neither
definition's public promotion flags changed.

The Qwen Image Edit qualification task is `5vMmdX_Dt3E4`. It used
`Qwen/Qwen-Image-Edit@ac7f9318f633fc4b5778c59367c8128225f1e3de`, two
inference steps, bfloat16, and model CPU offload. The generated 1024 x 1024
WebP has SHA-256
`ff3eb7aff49f6ec233df9fc79801732eaa18333b0121c31ebc824dbaa8e5008f`.
The receipt and review assets are under
`test-review/frontend-pipeline-e2e-2026-08-26/qwen-image-edit-cluster/`.
This remains a technical qualification output with `showcaseApproved: false`;
the definition's public promotion flags remain unchanged.

The Qwen Image image-to-image qualification task is `-0yiE5nwJYFp`. It used
`Qwen/Qwen-Image-2512@25468b98e3276ca6700de15c6628e51b7de54a26`, 256 x
256 output, two inference steps, strength 0.65, bfloat16, and model CPU
offload. The generated WebP has SHA-256
`8e720fd5d6434e9cf16fc868f8cc1a05fdd76e15298b737ccc57230e961ac399`.
The receipt and review assets are under
`test-review/frontend-pipeline-e2e-2026-08-26/qwen-image-to-image-cluster/`.
The output is technically valid but not showcase quality; publication remains
unchanged with `showcaseApproved: false`.

The Qwen Image inpainting qualification task is `kYyG7O1Qfhe3`. It used the
same pinned Qwen Image revision with a reviewed 1024px source/mask pair, saved
256 x 256 qualification dimensions, two inference steps, strength 0.8,
bfloat16, and model CPU offload. The generated WebP has SHA-256
`41fc729b3c72544b44c997126e0786e3c8d5714506d4dc87600bbb039729b2d4`.
The receipt and review assets are under
`test-review/frontend-pipeline-e2e-2026-08-26/qwen-image-inpaint-cluster/`.
The mask path executed correctly, but the aggressive qualification settings do
not produce showcase quality; publication remains unchanged.

The Qwen Image Edit inpainting qualification task is `d8CJn9ZGgH4Z`. It used
`Qwen/Qwen-Image-Edit@ac7f9318f633fc4b5778c59367c8128225f1e3de`, the
reviewed source/mask pair, two inference steps, strength 0.8, bfloat16, and
model CPU offload. The generated 1024 x 1024 WebP has SHA-256
`899c09030e89e8865d79c0d8a1b8712ba8f17b9a17f15b417605860bb2975538`.
The receipt and review assets are under
`test-review/frontend-pipeline-e2e-2026-08-26/qwen-image-edit-inpaint-cluster/`.
Visual inspection confirms that generation changed the masked object region
while preserving the surrounding room. This remains technical evidence with
`showcaseApproved: false`; publication flags remain unchanged.

The Qwen Image Edit Plus qualification tasks are `cDXJmv8V08SD` (single-image
edit) and `J33TZedDTeJy` (multi-reference edit). Both used
`Qwen/Qwen-Image-Edit-2511@6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`,
two inference steps, bfloat16, and model CPU offload. The generated WebP
SHA-256 values are
`ee56aca1068d563ae023dacb45ff1cbe62878665e53f1c02c7b28aad5afffead`
and `d43ca27b596d2c0171c4e84a633407dd5afca68f84daada7d0fa255825e585fa`.
The receipts and review assets are under
`test-review/frontend-pipeline-e2e-2026-08-26/qwen-image-edit-plus-cluster/`.
Both are valid technical executions; the two-step multi-reference result is not
semantically strong enough for showcase use. Both admissions remain unpublished
with `showcaseApproved: false`.

The Qwen Image Layered qualification task is `2LnilNHbwNy_`. It used
`Qwen/Qwen-Image-Layered@8f0ca708dfff6ba1dd5f2d85d78f8c108a040bcf`,
three requested layers at the recommended 640px source-resolution bucket, two
inference steps, bfloat16, and model CPU offload. The three generated RGBA WebPs
have SHA-256 values
`094df5951f7bb0b61211811948da73342d2f800879f6dcc8a862a84316fb47ca`,
`2ff6ca87fe1584e822b6eaea88da0d69969f8ce2fe3f87c0c68cd058ef4d0cf8`,
and `4d36d9cd518a048ca793c6e576efbff318a6921d703d3c29d7f6dff5c16bd20a`.
The receipt and review assets are under
`test-review/frontend-pipeline-e2e-2026-08-26/qwen-image-layered-cluster/`.
All three files retain alpha, but two-step decomposition quality is intentionally
technical rather than showcase-ready; publication remains unchanged.

The Qwen Image ControlNet qualification tasks are `dKMxhs33YBWr`
(text-to-image), `J06sUfHnFDc9` (image-to-image), and `PC4kdbNmLUYN`
(inpainting). All three used
`Qwen/Qwen-Image-2512@25468b98e3276ca6700de15c6628e51b7de54a26` with
`InstantX/Qwen-Image-ControlNet-Union@b13036f066d6dee7c20513e263d3d673055e9de8`,
256 x 256 output, two inference steps, a 0.9 conditioning scale, control range
0.1–0.85, maximum sequence length 512, bfloat16, and model CPU offload. The
generated WebP SHA-256 values are
`ec6b1fb45a8db3084318581eb09f66b9496b4dc8261939`,
`9f0482a6f2f3ca5f3727e4d24fe6a6c7df7e615653a4c9651822064a3c50546f`,
and `a87b2a447b9592155f2438bbe820b24187566cad82583be90e0ade3375310ee4`
in the same order. The combined receipt and review assets are under
`test-review/frontend-pipeline-e2e-2026-08-26/qwen-image-controlnet-clusters/`.
The browser test passed in 2.1 minutes and explicitly asserted that every saved
control value reached the corresponding expanded execution node. These are
technical low-step outputs, not showcase assets; all publication flags remain
unchanged with `showcaseApproved: false`.

The Wan 2.1 Modular text-to-video qualification task is `fWAsCv3UaxfW`. It
used
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers@0fad780a534b6463e45facd96134c9f345acfa5b`,
256 x 256 output, nine frames at 8 fps, two inference steps, maximum sequence
length 256, bfloat16, and model CPU offload. The generated H.264 MP4 contains
all nine requested frames, is 1.13 seconds long, and has SHA-256
`029df348e783756dc25e077f0ed9e54eb4b13e31c8db583257716ab544373835`.
The receipt, MP4, all extracted frames, and contact sheet are under
`test-review/frontend-pipeline-e2e-2026-08-26/wan-21-t2v-cluster/`. The
browser test passed in 1.2 minutes. The two-step output is useful technical
evidence but is not showcase quality; publication remains unchanged with
`showcaseApproved: false`.

The Wan 2.1 Modular first/last-frame-to-video qualification task is
`n4ArxD3deeaZ`. It used
`Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers@17c30769b1e0b5dcaa1799b117bf20a9c31f59d7`,
distinct 1024px source endpoints bound through the two separate expanded image
loader nodes, 256 x 256 output, nine frames at 8 fps, two inference steps,
maximum sequence length 256, bfloat16, and model CPU offload. The generated
H.264/yuv420p MP4 contains all nine requested frames, is 1.13 seconds long, and
has SHA-256
`bafd5c5ebe7b5f09aec4daa8eb38faff4d93dded4d26ce5d9666d4bd76f5a820`.
The browser test passed in 2.3 minutes. The receipt, MP4, all extracted frames,
contact sheet, and browser screenshot are under
`test-review/frontend-pipeline-e2e-2026-08-26/wan-21-flf-cluster/`. The
two-step endpoint interpolation is technically valid but intentionally not a
showcase asset; publication remains unchanged with `showcaseApproved: false`.

The Wan 2.1 Modular single-image-to-video qualification task is
`PYvxW-qjMTBN`. It used
`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers@b184e23a8a16b20f108f727c902e769e873ffc73`,
the saved source-image path, 256 x 256 output, nine frames at 8 fps, two
inference steps, maximum sequence length 256, bfloat16, and model CPU offload.
The generated H.264/yuv420p MP4 contains all nine requested frames, is 1.13
seconds long, and has SHA-256
`b4c5e35101a23216075019292734e6a8b8f3091e68c2e293eed53c68a089d3e1`.
The browser test passed in 2.4 minutes. Its exact receipt, MP4, all extracted
frames, contact sheet, Model Manager readiness screenshot, and generated-result
screenshot are under
`test-review/frontend-pipeline-e2e-2026-08-26/wan-21-i2v-cluster/`. This
two-step output is technical evidence rather than a showcase asset;
publication remains unchanged with `showcaseApproved: false`.

The ERNIE Image Turbo qualification task is `nttsH7oV8Esf`. It used
`baidu/ERNIE-Image-Turbo@bc68c81e2a1730a394d5fc9fae70713dee940140`,
the repository prompt enhancer, 1024 x 1024 output, one technical inference
step, guidance 1, bfloat16, model CPU offload, and seed 54001. The generated
WebP has SHA-256
`8d9b3f9344f916d4acfbbfcbaef89ad094399fa3d4cb721d9b9a6281b3043c08`.
The browser test passed in 55.9 minutes including the 31.65 GB visible download;
model execution itself loaded all seven pipeline components and completed its
denoise step in about 39 seconds. The receipt, generated WebP, generated-result
screenshot, download-start screenshot, and Model Manager Ready screenshot are
under `test-review/frontend-pipeline-e2e-2026-08-26/ernie-image-cluster/`.
This remains qualification evidence with `showcaseApproved: false`; no public
promotion flag changed.

## 2026-08-28 admitted-route and shared-runtime closure

The priority order for this tranche is now satisfied: shared runtime/resource
handling first, the strengthened admitted Diffusers routes second, and
contract-only family expansion only after those gates.

### Shared runtime/resource handling

- [x] Model Manager cache deletion now releases app-owned node/model/component
      caches and trims the accelerator allocator before deleting immutable Hub
      snapshots. Its response contains a validated `runtimeRelease` receipt.
- [x] The visible Model Manager flow deleted the pinned Wan 2.2 text-to-video
      snapshot and reclaimed `126199197011` bytes while preserving saved
      workflows and canonical definitions.
- [x] The same flow deleted the pinned Wan 2.2 image-to-video snapshot and
      reclaimed `126202473720` bytes. It released six nodes and one model,
      trimmed the allocator, reported no cleanup errors, left zero deleted open
      file descriptors, and returned the machine to about 243 GiB free.
- [x] Synchronous managed-graph signal requests now dispatch directly on the
      websocket event loop instead of waiting behind ordinary progress and
      dynamic-definition broadcasts. The future is created and cleaned up on
      its owning loop, and the browser-reconciliation wait remains bounded at
      60 seconds. This fixes the repeatable third-workflow SDXL Guider timeout
      after Save/refresh transitions.
- [x] Focused signal coverage passes (`22` server-security tests plus `29`
      subtests; `18` Modular Guider/Scheduler/Layers tests plus `31` subtests).

The turnover receipts and screenshots are under
`test-review/frontend-pipeline-e2e-2026-08-27/storage-turnover/`, including the
definition-specific files
`model-manager-eviction-wan22-text-to-video-result.json` and
`model-manager-eviction-wan22-image-to-video-result.json`.

### Strengthened admitted Diffusers lifecycle

- [x] All `14/14` routes in the remaining admitted tranche now pass the exact
      frontend lifecycle: insert Cluster Node, edit nondefault parameters,
      Save, browser refresh, compare persisted parameters, expand, compare the
      collapsed/expanded graph, execute, and retain a hashed generated asset.
- [x] All nine SDXL combinations pass together as one serial browser test in
      7.2 minutes after the signal-lifecycle fix. The exact task IDs are
      `1oo2yRmuVQRb`, `zu_XOTPQQx5O`, `Ohta6ulgLOeZ`, `7A1obEnWD1-x`,
      `ElhHerIrgqtM`, `82tENa1h78ri`, `R_0Q3_RrKIMR`, `qw5dQ0O2oap0`, and
      `14C5w2a19hbX`, in the scenario order stored by the combined receipt.
- [x] LTX text-to-video, LTX image-to-video, and LTX-2 image-to-video retain
      their strengthened frontend lifecycle evidence from tasks
      `5cpBN6QNPIan`, `_R2l1dv0tmme`, and `tKvuz8NMqo4i`.
- [x] Wan 2.2 text-to-video passed task `GgQplt9m5jUl` at pinned revision
      `5be7df9619b54f4e2667b2755bc6a756675b5cd7`; its generated MP4 hash is
      `sha256:bytes:36cbc4f04c29fa129d8c21fc7c24dadb75ca46cab34958b83c660d75b551f44b`.
- [x] Wan 2.2 image-to-video passed task `NOWutT3Wu4jJ` at pinned revision
      `596658fd9ca6b7b71d5057529bbf319ecbc61d74`; before/after parameter
      snapshots are identical, graph equivalence is true, and its generated
      MP4 hash is
      `sha256:bytes:e37f9b549665887cbdb724ae2bef3bfe2467062afc0502d63d7e346f61d3d93b`.

The SDXL combined receipt and nine generated assets are under
`test-review/frontend-pipeline-e2e-2026-08-27/sdxl-remaining-clusters/`. Wan
receipts and MP4s are under
`test-review/frontend-pipeline-e2e-2026-08-27/remaining-admitted-video-clusters/`.

### Post-change release gates

- [x] Full backend after the Helios tranche: `2219 passed`, `53 skipped`,
      `6706 subtests passed`.
- [x] Complete client `npm run check`: lock/license/format/lint/typecheck/style,
      unit tests, production build, and bundle budget all pass.
- [x] Complete mocked Studio browser suite: `114/114 passed`.
- [x] The live SDXL nine-route serial browser suite passes after a fresh backend
      restart, proving the multi-workflow signal fix under the original failure
      sequence.

### Remaining after admitted-route closure

- [ ] Keep all new technical outputs unpublished until showcase/output approval
      and definition-specific promotion receipts exist; `executable`,
      `liveProof`, and `autoEligible` public flags remain unchanged.
- [ ] Qualify the remaining `30` contract-only workflows across the `11`
      documented image, video, and multimodal families, handling each exact
      artifact/license/ROCm gate before promotion.
- [ ] Generate improved showcase-quality assets after implementation breadth is
      complete; current low-step outputs are qualification evidence only.

## 2026-08-28 contract-only expansion — LTX-2 condition tranche

- [x] Reviewed the current official Diffusers LTX-2 API and pinned source before
      implementation. `LTX2AutoBlocks` selects `condition` from `conditions +
      prompt`; its exact block order is prompt enhancement, text encoding,
      duration, condition encoding, denoising, and condition-aware decoding.
- [x] Promoted `LTX2ModularPipeline:condition` through the exact pinned
      `LTX2ConditionPipeline` equivalent route without changing the upstream
      block hierarchy or publishing a split-block execution claim.
- [x] Added a backend-owned `conditionImages -> conditions` adapter contract.
      The collapsed Cluster renders it as a real multi-image picker, persists
      the ordered paths in the workflow instance, and materializes them through
      `modules.Image.Load`; the generator converts the images to official
      `LTX2VideoCondition` instances and distributes them from the first through
      last generated frame. Condition strength remains editable and durable in
      the expanded execution graph.
- [x] Added the exact Studio spec, task-template media contract, immutable base
      artifact/profile seal, static admission, and client media-control
      projection. Public execution, live-proof, Auto, and showcase flags remain
      disabled.
- [x] Focused verification passes: `117` backend tests plus `1645` subtests;
      Hugging Face Cluster client suite `28/28`; client TypeScript build passes.
- [ ] Run the strengthened visible lifecycle and real generation after the
      exact LTX-2 snapshot has been restored through Model Manager. Preserve the
      receipt, MP4, extracted frames, contact sheet, and screenshots under the
      contract-only review directory.
- [x] Implement `LTX2ModularPipeline:in_context` as a distinct route. Official
      Diffusers requires `LTX2ReferenceCondition` values and an IC-LoRA loaded
      on `LTX2InContextPipeline`; ordinary condition execution must never be
      relabeled as in-context. The reviewed first candidate is
      `Lightricks/LTX-2-19b-IC-LoRA-Canny-Control@28be96236294914042a1605f37e3f7812b45f43f`
      with weight `ltx-2-19b-ic-lora-canny-control.safetensors` (654,465,256
      bytes). The exact loader, reference-video preprocessing graph, bindings,
      task contract, static admission, and synchronized video/audio executor
      are implemented. Fixture execution proves both immutable snapshots,
      named safe LoRA loading, reference-condition construction, frame-count
      propagation, downscale/attention controls, and video/audio output.
- [x] The expanded focused regression passes: `269` backend tests, `14`
      expected skips, and `1226` subtests; the Hugging Face Cluster client suite
      passes `28/28`, and client TypeScript compilation passes.
- [ ] Run its strengthened visible lifecycle and real generation after the
      derivative revision's license is acknowledged and the exact artifacts
      are installed. This is a legal/artifact gate, not an implementation
      substitute.

The remaining contract-only count after the LTX-2 tranche was `43` workflows. All four LTX-2
Modular workflows have reviewed equivalent standard routes; the two newly
added routes remain unpublished until their visible lifecycle evidence and
output review are complete.

## 2026-08-28 contract-only expansion — FLUX.2 dev tranche

- [x] Reviewed the current official Diffusers FLUX.2 pipeline API, the pinned
      `Flux2AutoBlocks` source, and the immutable Hub repository before
      implementation. The two upstream workflows overlap substantially:
      `text2image` and `image_conditioned` share the text encoder, VAE path,
      denoiser, and decoder; the latter activates image preprocessing/encoding
      and image-latent preparation.
- [x] Added exact equivalent routes for
      `Flux2ModularPipeline:text2image` and
      `Flux2ModularPipeline:image_conditioned` through the official
      `Flux2Pipeline`. The image-conditioned Cluster keeps one-or-more ordered
      references and maps to `multi_image_reference_edit`; it is not reduced
      to a one-image-only route.
- [x] Pinned `black-forest-labs/FLUX.2-dev` at
      `26afe3a78bb242c0a8bb181dcc8937bb16e5c66c`, recorded the exact 35-file
      component closure (`112,823,045,100` bytes), added the standard generic
      loader/generate/edit contracts, resource envelope, task contracts,
      equivalent Cluster specs, and static admissions.
- [x] Removed the stale duplicate “contract-only” publication for both the
      now-closed LTX-2 and FLUX.2 families. The pinned upstream ledger now
      classifies their modular classes as reviewed equivalents and their exact
      standard executors as executable graph surfaces; public/live-proof flags
      remain disabled.
- [x] Combined focused verification passes: `394` backend tests, `18` expected
      skips, and `1655` subtests. This includes generic FLUX.2 text and ordered
      multi-reference fixture execution plus the full LTX-2 video/audio slice.
- [x] Fresh-process verification passes after restarting the complete backend:
      the live API exposes `240` Studio execution specs and `102` Hugging Face
      definitions with `62/62` static admissions accepted. Both FLUX.2
      definitions resolve to their exact equivalent-standard modes, neither
      FLUX.2 nor LTX-2 is duplicated as an experimental capability, the client
      Hugging Face library suite passes `28/28`, and TypeScript compilation is
      clean.
- [ ] Obtain exact FLUX.2 license acceptance for the pinned revision before
      downloading the gated 112.8 GB closure. Then run the strengthened visible
      lifecycle for both Clusters and preserve generated assets/receipts.

The remaining contract-only count is now `41` workflows across `15` families.

## 2026-08-28 contract-only expansion — Anima tranche

- [x] Reviewed the official Anima Modular Diffusers source, tests, conversion
      script, immutable Hub model index, and license before implementation. The
      authoritative workflows overlap substantially: `text2image` is
      `text_encoder -> denoise -> decode`; `img2img` reuses those stages and
      inserts `vae_encoder`, plus the image-aware denoise inputs.
- [x] Added native package-owned whole-workflow execution for both official
      routes through `AnimaModularPipeline` and `AnimaAutoBlocks`. The graph
      uses sealed `PipelineState` hand-offs between reviewed text encode, image
      encode, denoise, and image decode nodes; it does not relabel a standard
      full-pipeline call as split Modular execution.
- [x] Pinned
      `circlestone-labs/Anima-Base-v1.0-Diffusers@073c3a9db359c31ad0e8aa268d15775473c2176c`,
      sealed its 16-file selected closure (`5,642,032,794` bytes), four
      safetensors weight digests (`5,628,156,702` bytes), Modular index hash,
      seven component identities, and both official workflow/action maps in
      `data/anima-artifact-review.json`.
- [x] Added the exact Studio profiles, task contracts, static admissions,
      resource limits, loader registration, model artifact catalog entry, and
      upstream coverage classification. The live backend now publishes `242`
      Studio specs, `102` Cluster definitions, `505` upstream block contracts,
      and both Anima routes at the exact revision.
- [x] Added a single revision-bound frontend acknowledgement used by Model
      Manager and Studio Run. The reviewed license limits model/derivative use
      to non-commercial purposes, permits commercial output use subject to its
      restrictions, and prohibits using outputs to train a competing model;
      the UI summary preserves that distinction.
- [x] Fixed live client contract gaps exposed by the fresh browser: generic
      whole-workflow binding sources, LTX-2 `in_context_to_video` mode and
      execution roles, and the `conditionImages` task-media field.
- [x] Focused backend verification passes: `136 passed`, `7` expected skips,
      and `1067` subtests. Client TypeScript compilation, the `28/28` Cluster
      library suite, the full live catalog insertion test, and the new Anima
      insert/expand/edit/Save/refresh/collapsed-expanded materialization-parity
      browser test all pass.
- [ ] Download and run the exact model through Model Manager only after the
      user acknowledges this exact CircleStone license/revision. Then retain
      the generated image, runtime receipt, and screenshots and request output
      approval before changing `executable`, `liveProof`, or `autoEligible`.

Primary implementation references:

- `https://github.com/huggingface/diffusers/blob/2f7e0154a9db246e95c9ede43edba7db5b130805/src/diffusers/modular_pipelines/anima/modular_blocks_anima.py`
- `https://github.com/huggingface/diffusers/blob/2f7e0154a9db246e95c9ede43edba7db5b130805/tests/modular_pipelines/anima/test_modular_pipeline_anima.py`
- `https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers/tree/073c3a9db359c31ad0e8aa268d15775473c2176c`
- `https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers/blob/073c3a9db359c31ad0e8aa268d15775473c2176c/LICENSE.md`

The remaining contract-only count is now `39` workflows across `14` families.

## 2026-08-28 contract-only expansion — Helios tranche

- [x] Reviewed the three official pinned Modular Diffusers implementations,
      their shared encoders/denoisers/decoders, immutable Hub repositories,
      model indexes, selected component closures, and Apache-2.0 metadata
      before implementing their graph routes.
- [x] Added native package-owned execution for all nine official workflows:
      text-to-video, image-to-video, and video-to-video for
      `HeliosModularPipeline`, `HeliosPyramidModularPipeline`, and
      `HeliosPyramidDistilledModularPipeline`. The routes share the reviewed
      text encoder, conditional VAE encoder, denoiser, and video decoder nodes;
      pyramid stage schedules and distilled guidance behavior remain
      pipeline-specific.
- [x] Added exact revision-pinned artifact records for
      `BestWishYsh/Helios-Base@5c50b6bc90eae9bd815d2a50b0c9877e3fd2cf88`,
      `BestWishYsh/Helios-Mid@477c55427ec0ea774bdebd0fbe736313cfc5a312`,
      and
      `BestWishYsh/Helios-Distilled@b991c0379a018f4de3227d95468237f56066f5bb`.
      Each selected closure has 24 files and 80,481,086,028 bytes of model
      weights; exact selected-closure sizes are recorded in
      `data/helios-artifact-review.json`.
- [x] Added the nine exact Studio execution specs, task/media contracts,
      component-revision propagation, static admissions, resource/offload
      profiles, frontend capability profiles, graph roles, and managed-control
      policy. All `65/65` reviewed Diffusers candidates now pass static graph
      admission.
- [x] Backend node/fixture and ledger verification passes: `110` tests, one
      expected skip, and `981` subtests. The fixtures exercise base,
      pyramid-stage, distilled, image-conditioned, video-conditioned, and
      video-decoding behavior without substituting a generic full-pipeline
      executor.
- [x] Complete the fresh-backend visible browser proof for Helios Pyramid
      Distilled: insert, expand, edit prompt and generation parameters, Save,
      refresh, compare the exact overrides, prove collapsed/expanded canonical
      materialization parity, and verify the compiled native action sequence.
      The proof passes against a freshly restarted backend and preserves its
      screenshots under the dedicated Helios review-assets directory.
- [ ] Run remote heavy-hardware generation and preserve the video, runtime
      receipt, frames, contact sheet, and screenshots before enabling public
      executable/live-proof/Auto flags. Local real generation is not claimed:
      each selected closure is about 80.5 GB and still requires a qualified
      memory/offload environment.

Primary implementation references:

- `https://github.com/huggingface/diffusers/blob/2f7e0154a9db246e95c9ede43edba7db5b130805/src/diffusers/modular_pipelines/helios/modular_blocks_helios.py`
- `https://github.com/huggingface/diffusers/blob/2f7e0154a9db246e95c9ede43edba7db5b130805/src/diffusers/modular_pipelines/helios/modular_blocks_helios_pyramid.py`
- `https://github.com/huggingface/diffusers/blob/2f7e0154a9db246e95c9ede43edba7db5b130805/src/diffusers/modular_pipelines/helios/modular_blocks_helios_pyramid_distilled.py`
- `https://huggingface.co/BestWishYsh/Helios-Base/tree/5c50b6bc90eae9bd815d2a50b0c9877e3fd2cf88`
- `https://huggingface.co/BestWishYsh/Helios-Mid/tree/477c55427ec0ea774bdebd0fbe736313cfc5a312`
- `https://huggingface.co/BestWishYsh/Helios-Distilled/tree/b991c0379a018f4de3227d95468237f56066f5bb`

At this historical checkpoint, the remaining contract-only count was `30`
workflows across `11` families. The Wan Animate 2 tranche below reduces the
current count to `28` workflows across `9` families.

The fresh-browser Helios lifecycle evidence is retained at:

- `/home/sayak/MoDiff/review-assets/diffusers-contract-expansion/helios-2026-08-28/diffusers-helios-structural-cluster-expanded.png`
- `/home/sayak/MoDiff/review-assets/diffusers-contract-expansion/helios-2026-08-28/diffusers-helios-cluster-save-refresh.png`

The final clean-process smoke also reran the existing Anima lifecycle alongside
Helios; both tests passed (`2/2`) against the newly restarted backend.

## Next contract-only tranche — Wan Animate 2

- [x] Confirmed against the pinned official Diffusers source that base and
      distilled each expose one six-stage Modular workflow:
      `text_encoder -> image_encoder -> video_encoder -> vae_encoder -> denoise
      -> decode`. Both share the same semantic stages; the distilled denoise
      inner loop and scheduler/default-step contract remain distinct.
- [x] Reconfirmed both immutable public repositories and their existing exact
      artifact reviews. The base revision is
      `7d48412d7b903ff3a89f4f5a960d99e1899605a1`; the distilled revision is
      `59e4141466bcb1bf9733eca1bc78be6891c9fbdf`. Each selected weight inventory
      is 45,920,934,868 bytes.
- [x] Replace mutable/null nested component descriptors with the exact selected
      top-level revision during loading, while rejecting cross-repository
      descriptors and remote Python. Dedicated tests prove that both the null
      and `refs/pr/*` same-repository descriptors resolve to the exact reviewed
      40-character top-level commit.
- [x] Add the six sealed state-handoff actions, exact image/driving-video
      bindings, base/distilled denoise controls, decode/export graph, task
      contract, capability/resource profiles, static admissions, and fixture
      execution. Base preserves the official 40-step/guidance-3 recipe;
      Distilled is presented with the official 10-step/guidance-1 recipe while
      still allowing an explicit persisted step override.
- [x] Run the same insert/edit/Save/refresh/materialization-parity browser
      lifecycle. Keep real execution, public flags, Gallery, and Auto disabled
      until compiled Flex Attention is qualified on suitable remote hardware
      and the generated video is reviewed.

The fresh visible-frontend proof uses the Distilled Cluster and verifies
library insertion, all six official stage paths, an internal prompt edit,
reference-prompt/segment/previous-conditioning-frame edits,
width/height/step/FPS/guidance/seed overrides, collapsed/expanded canonical
graph parity, backend workflow Save, browser refresh, and byte-equivalent
restoration. The semantic edits are asserted on the materialized execution
nodes, so the proof rejects a control that merely persists visually without
reaching the official block call. Evidence is retained at:

- `/home/sayak/MoDiff/MoDiff/data/qualification/local-review/hugging-face-clusters/wan-animate-2-cluster/save-refresh-expanded.png`
- `/home/sayak/MoDiff/MoDiff/data/qualification/local-review/hugging-face-clusters/wan-animate-2-cluster/frontend-persistence-result.json`

The upstream coverage ledger now classifies `133` Diffusers pipeline symbols
as executable and `9` as contract-only. At the Wan checkpoint, the remaining
structural expansion was `28` workflows across `9` families. The sealed
HunyuanVideo 1.5, Stable Diffusion 3, and Krea 2 adapter tranches below reduce
that implementation queue to `21` workflows across `4` families: Cosmos 3
Distilled, Cosmos 3 Omni, MiniMax H3, and LTX-2.5.

Real Wan Animate 2 Distilled generation is now proven through the visible
frontend. Model Manager installed the exact
`Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers@59e4141466bcb1bf9733eca1bc78be6891c9fbdf`
snapshot from the reviewed 30-file, 45,947,569,675-byte request and reports the
cache complete with no missing or corrupt files. A fresh-browser repair also
proved that a sealed optional-runtime integrity failure is recoverable through
Setup, activates the replacement overlay, and survives the controlled backend
restart. Model Manager now reconciles authoritative background-download state
while open, so a browser that misses the final long-request transition cannot
remain indefinitely at `Installing` after the backend has completed.

The generation proof inserts the Distilled Cluster from the library, changes
the prompt, reference prompt, media, dimensions, segment length, previous-frame
conditioning, step count, FPS, guidance, and seed, saves the backend workflow,
refreshes the browser, and verifies exact restoration. It then materializes the
six official stages plus media load/export nodes and submits the graph through
the frontend. The exact 14B components load with model CPU offload, the required
`transformer.compile_repeated_blocks(fullgraph=False)` path compiles, and one
nine-frame segment executes on the Radeon 8060S. The qualification MP4 is H.264,
336x192, 16 FPS, 9 frames, and 13,590 bytes with SHA-256
`d63a732af7b80c50f92b4a63a45417b316f6564279be00df7addd87175d773dc`.
This output is a functional qualification artifact, not a showcase-approved
promotion asset.

Execution also exposed and closed one upstream contract mismatch without
weakening validation: official Wan Animate 2 blocks declare `AutoTokenizer`
and `SchedulerMixin`, while the pinned official Hub indexes serialize the
concrete `T5TokenizerFast` and `FlowMatchEulerDiscreteScheduler`. MoDiff now
admits only those exact repository/component aliases for the base and
Distilled repositories; focused tests accept both official indexes and reject
an alternate tokenizer class.

The isolated prerequisite smoke is green on this host: PyTorch
`2.9.1+rocm7.2.0.git7e1940d4` compiled and executed a BF16 Flex Attention call
with a compiled block mask on the Radeon 8060S in 5.899 seconds. Its receipt is
retained at
`data/qualification/local-review/hugging-face-clusters/wan-animate-2-distilled-live/flex-attention-smoke.json`.
This proves the kernel/backend capability, but it is not a substitute for the
full 14B model's memory/offload and generated-video proof.

Completion gates for this tranche:

- focused backend after the semantic binding additions: `100 passed`, `882`
  subtests passed;
- full backend after the required compile policy: `2,225 passed`, `53 skipped`,
  `6,726 subtests passed`;
- complete client `npm run check`: formatting, lint, typecheck, unit tests,
  production build, and bundle budget all passed;
- strengthened visible live-backend Wan lifecycle: `1/1 passed` after a fresh
  app restart;
- required regional-compile loader policy: `64 passed`, `12` subtests passed
  across the focused offload and Wan artifact-review suites;
- exact visible Model Manager installation: `1/1 passed` after the real
  download, cache verification, optional-runtime repair/activation, and backend
  restart;
- exact visible frontend generation: `1/1 passed` in 2.1 minutes, including
  insert/edit/Save/refresh/materialization and the real nine-frame model run;
- pinned Wan component-alias validation: `2 passed`, `9` subtests passed with
  the sealed Transformers overlay active;
- complete client `npm run check` after download reconciliation and the live
  generation fixes: passed, including formatting, lint, typecheck, unit tests,
  production build, and bundle budget (`332,596` entry gzip bytes and `591,802`
  total gzip bytes);
- full backend release gate after the component-alias change: `2,225 passed`,
  `54 skipped`, `6,726` subtests passed in 123.33 seconds;
- the generated upstream/template/Comfy provenance ledgers pass their offline
  deterministic validation after the two-symbol promotion.

Primary implementation references:

- `https://github.com/huggingface/diffusers/blob/2f7e0154a9db246e95c9ede43edba7db5b130805/src/diffusers/modular_pipelines/wan_animate_2/modular_blocks_wan_animate_2.py`
- `https://github.com/huggingface/diffusers/blob/2f7e0154a9db246e95c9ede43edba7db5b130805/src/diffusers/modular_pipelines/wan_animate_2/modular_blocks_wan_animate_2_distilled.py`
- `https://github.com/huggingface/diffusers/blob/2f7e0154a9db246e95c9ede43edba7db5b130805/src/diffusers/modular_pipelines/wan_animate_2/denoise.py`
- `https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan_animate_2`
- `https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers/commit/ff28d41a9c0c331137691847cc8700593d785593`

Evidence is retained at:

- `data/qualification/local-review/hugging-face-clusters/wan-animate-2-distilled-live/frontend-install-result.json`
- `data/qualification/local-review/hugging-face-clusters/wan-animate-2-distilled-live/frontend-generation-result.json`
- `data/qualification/local-review/hugging-face-clusters/wan-animate-2-distilled-live/output-validation.json`
- `data/qualification/local-review/hugging-face-clusters/wan-animate-2-distilled-live/wan-animate-2-distilled-generated.mp4`
- `data/qualification/local-review/hugging-face-clusters/wan-animate-2-distilled-live/wan-animate-2-distilled-contact-sheet.png`

Remaining Wan promotion gate: review/approve a showcase-quality output, then
write the definition-specific promotion receipt before enabling public
`executable`, `liveProof`, Gallery, or Auto flags. The bounded one-step output
is deliberately not promoted.

## Following Wan — HunyuanVideo 1.5 research gate

- [x] Revalidated the pinned package structure without promoting it: the
      `text2video` workflow is `text_encoder -> denoise -> decode`; the
      `image2video` workflow is `text_encoder -> vae_encoder -> image_encoder
      -> denoise -> decode`.
- [x] Revalidated the two exact public Diffusers artifacts already recorded in
      `data/hunyuanvideo-1.5-artifact-review.json`: T2V revision
      `286be7ce72277246578a3e3cc2487e95ddae5bcf` and step-distilled I2V revision
      `854c04a4c8a53d990b418c7478f0802c0fc8c726`.
- [x] Added package-owned sealed block adapters for the exact official stage
      overlap. Text-to-video materializes text encoding, denoising, and decode;
      image-to-video preserves the separate VAE and SigLIP image encoders
      before its MeanFlow-aware denoiser and decoder. Focused fixture execution
      and library-contract coverage pass (`53 passed`, `7 skipped`, `83`
      subtests passed) without loading or exposing model weights.
- [x] After the active Wan Animate 2 base Model Manager installation finishes,
      restart the backend and run a fresh-browser Hunyuan structural lifecycle:
      insert, expand, edit, Save, refresh, and verify exact restoration. The
      browser must continue to show no Prepare/Run/download claim for either
      Hunyuan route.
- [ ] Do not expose either route as executable or downloadable until the
      publisher's conflicting territory clauses receive an explicit legal
      disposition. Implementation may prepare sealed adapters and fixtures,
      but it must not silently turn a contract-only entry into a runnable
      artifact.

## Wan Animate 2 base live qualification

- [x] Generalized the visible-frontend Model Manager and generation tests so
      the same lifecycle can select either the reviewed base or Distilled
      definition without duplicating a weaker test path.
- [x] Install the exact base snapshot
      `Wan-AI/Wan2.2-Animate-2-14B-Diffusers@7d48412d7b903ff3a89f4f5a960d99e1899605a1`
      through Model Manager and retain its install receipt/screenshots. All 30
      reviewed files are present at the exact commit; the cache contains
      `45,947,570,056` bytes of immutable blobs.
- [x] Run the base Cluster through the visible frontend: insert, edit internal
      controls, Save, refresh, verify the restored values and materialized
      six-stage graph, then generate and validate the qualification MP4.
- [ ] Keep the generated base asset qualification-only until a separate
      showcase-quality output is reviewed and a definition-specific promotion
      receipt is approved.

The live base lifecycle passed `1/1` in 1.8 minutes. It restored the exact
parameter/execution overrides after Save and browser refresh, materialized the
reviewed loader, media inputs, four official Modular stages, and video export,
then generated a real nine-frame H.264 MP4. The pinned base repository exposed
one variant-specific concrete component alias not shared by Distilled:
`DPMSolverMultistepScheduler` for base versus
`FlowMatchEulerDiscreteScheduler` for Distilled. Validation now admits each
concrete scheduler only for its exact repository and continues to reject the
opposite scheduler; the focused loader proof passes with `1 passed`, `6`
subtests passed.

Base evidence is retained at:

- `data/qualification/local-review/hugging-face-clusters/wan-animate-2-base-live/frontend-install-result.json`
- `data/qualification/local-review/hugging-face-clusters/wan-animate-2-base-live/frontend-generation-result.json`
- `data/qualification/local-review/hugging-face-clusters/wan-animate-2-base-live/wan-animate-2-base-live-generated.mp4`
- `data/qualification/local-review/hugging-face-clusters/wan-animate-2-base-live/generated-frame-00.png`
  through `generated-frame-08.png`

The generated MP4 is 336x192 at 16 FPS, 0.563 seconds, 13,245 bytes, with
SHA-256 `6538b2ab7a07e44839acc24027e9970c3d35fc075d8aa8edafc524a8c805f953`.
It remains qualification-only because the run deliberately used one denoising
step and was not intended as a showcase asset.

## Stable Diffusion 3 sealed structural tranche

- [x] Revalidated the pinned official Modular Diffusers source. Text-to-image
      uses `text_encoder -> denoise -> decode`; image-to-image adds the
      separately inspectable `vae_encoder` before the shared denoise and decode
      stages. The VAE and denoise containers retain their official nested
      preprocess, timestep, latent-preparation, and loop blocks in the expanded
      Cluster hierarchy.
- [x] Added exact package-owned adapters for the three CLIP/T5 prompt encoders,
      image preprocessing/VAE encoding, T2I/I2I denoising, CFG guider update,
      strength, seed, and PIL decode. The combined Hunyuan/SD3 focused gate is
      green (`61 passed`, `7 skipped`, `83` subtests passed).
- [x] After the active Wan base download completes, run the fresh-browser SD3
      insert/expand/edit/Save/refresh proof alongside Hunyuan. SD3 must remain
      catalog-only with no Prepare/Run or Model Manager download claim.
- [ ] Do not download or advertise the SD3 Medium artifact until the Stability
      noncommercial gate has explicit task-scoped product/legal acceptance.
      Existing artifact tests continue to require zero catalog exposure and
      zero model-weight download.

## Krea 2 sealed structural tranche

- [x] Revalidated base and Turbo against the pinned official Modular Diffusers
      source. Both use `text_encoder -> denoise -> decode` and share the
      Qwen3-VL text architecture, packed-latent preparation, position IDs, and
      Qwen-Image VAE decode. Their exact controls remain distinct: base exposes
      symmetric CFG, negative prompt, 28 steps, and guidance 4.5; Turbo exposes
      no negative prompt or guider and defaults to 8 steps.
- [x] Added separate base/Turbo text and denoise actions plus the shared exact
      decoder. Focused fixture tests reject cross-variant execution and prove
      the distinct calls/defaults. The expanded gate is green (`68 passed`, `7`
      skipped, `87` subtests passed).
- [x] Run the clean-browser all-definition insertion smoke after the Wan base
      download completes and the backend is restarted. Krea base and Turbo must
      remain catalog-only and expose neither Prepare/Run nor Model Manager
      download controls.
- [ ] Keep both gated model artifacts unavailable until their custom license,
      moving acceptable-use policy, revenue threshold, and deployment content
      filters receive explicit task-scoped legal/product approval.

## Ideogram 4 sealed structural tranche

- [x] Revalidated the pinned official four-stage Modular route:
      `prompt_upsample -> text_encoder -> denoise -> decode`. Expanded Clusters
      retain the six internal denoise preparation/loop stages, including both
      conditional and unconditional transformers and the after-denoise
      unpatchification step.
- [x] Added separate package-owned actions for optional local prompt
      upsampling, Qwen3-VL text encoding, asymmetric-CFG denoising, and Flux2
      VAE decode. The visual denoise node accepts an exact JSON guidance
      schedule only when it has one finite value per inference step; an empty
      value preserves the official upstream schedule. The focused gate is green
      (`74 passed`, `7 skipped`, `87` subtests passed).
- [x] Verify insertion/expansion in the clean-browser all-definition smoke
      after the Wan base download and backend restart. The route must remain
      catalog-only with no Prepare/Run or model download action.
- [ ] Keep the model artifacts and optional prompt-enhancer artifact closed:
      the publisher gate/noncommercial terms have not been accepted, exact
      authenticated artifact identities remain unresolved, and MoDiff does not
      permit fallback to the hosted Ideogram prompt-rewrite API.

## Structural Cluster graph catalog completion — 2026-08-28

- [x] Added exact package-owned graph adapters for all four Cosmos 3 Distilled
      workflows: text-to-image, text-to-video, image-to-video, and
      video-to-video. The conditioned routes retain the separate VAE encoder;
      the common text-encode, denoise, and decode stages remain independently
      visible after expansion.
- [x] Added exact package-owned graph adapters for all ten Cosmos 3 Omni
      workflows, including sound-generating and action-policy/forward/inverse
      dynamics routes. The official after-decode stage is a distinct visual
      node. Action conditions are constructed only inside the optional runtime;
      importing the catalog never executes repository or model code.
- [x] Added exact package-owned graph adapters for the three MiniMax H3
      workflows. FL2VA and Ref2VA preserve before-encode, text-encode,
      VAE-encode, denoise, and joint video/audio decode as separate stages.
- [x] Added exact package-owned graph adapters for all four LTX-2.5 workflows:
      text-to-video, image-to-video, condition, and in-context. Duration,
      condition encoding, reference encoding, native diffusion decoding, and
      audio sample-rate propagation follow the reviewed upstream contracts.
      Prompt enhancement is not inserted implicitly because the official
      default workflow disables it; it remains an explicit composable block.
- [x] The current first-party library now has sealed graph adapters for all
      `102/102` definitions: `94` Diffusers Cluster definitions and `8`
      Transformers Cluster definitions. This closes the structural insertion,
      expansion, and parameter-persistence implementation queue; it does not
      claim that gated or uninstalled artifacts are executable.
- [x] Focused backend coverage after the final structural tranche is green:
      `61 passed`, `7 skipped`, `83 subtests passed`. Client TypeScript
      validation is also green after adding the before/after encode, duration,
      condition-encode, and reference-encode graph roles.
- [x] Restart the app after the Wan Animate 2 base qualification so the visible
      frontend loads the complete catalog, then run clean-browser insertion and
      save/refresh persistence coverage. Gated definitions must continue to
      expose no Prepare, Run, or Model Manager installation claim.
- [x] Complete the post-catalog release regression. The canonical backend suite
      passes `2,243` tests with `54` skipped and `6,726` subtests passed. The
      complete client `npm run check` gate passes formatting, linting,
      TypeScript, style, unit, production-build, and bundle-budget checks. The
      complete mocked Studio browser suite passes `114/114` scenarios in 6.1
      minutes, including Cluster/User Node persistence, Model Manager, Auto and
      Expert execution forms, and the speech-to-text workflows.
- [x] Prove clean-browser structural lifecycle behavior for all `94` reviewed
      Diffusers definitions, plus strengthened save/refresh and complete
      upstream AutoBlocks-tree checks for HunyuanVideo 1.5 and Stable Diffusion
      3. These policy-gated definitions remained non-executable throughout the
      browser tests.
- [ ] Keep execution promotion separate from structural coverage. Cosmos 3,
      MiniMax H3, LTX-2.5, HunyuanVideo 1.5, SD3, Krea 2, and Ideogram 4 remain
      closed until their exact artifact, runtime, policy, and legal gates are
      individually satisfied and their live outputs are reviewed.
