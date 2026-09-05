# MoDiff Generic Nodes, Upstream Coverage, and Asset Completion Plan

Status date: 2026-08-20

This document is the handoff source of truth for continuing the MoDiff campaign in a new chat. It separates source/graph readiness from real execution, asset quality, rights, hardware qualification, and release eligibility.

## Primary goals

1. Keep the Studio client model-neutral. It must render backend-published contracts; it must not decide behavior from Diffusers or Transformers model names.
2. Review every relevant surface in the latest pinned Hugging Face Diffusers and Transformers commits. Every upstream class or finite task semantic must have exactly one evidence-backed disposition: executable, equivalent, contract-only, research-blocked, or intentionally excluded.
3. Provide generic nodes, canonical workflows, and authoring/template contracts for every safely supportable task.
4. Generate review assets as early as possible without allowing generation work to block source work.
5. Treat every failed template run as a defect to diagnose, fix, test, and rerun. Never turn failures into skips or silently omit them.
6. Make workflows portable through explicit platform/model fallback policies for quantization, offload, attention, dtype, and device placement. Do not claim universal FlashAttention or universal quantization.
7. Preserve previously approved visual examples. Re-run only examples whose current graph, prompt, artifact, runtime, or numerical contract cannot be reconciled to the approved evidence.

## Non-negotiable operating rules

- Read both repository `AGENTS.md` files before editing.
- Preserve the inherited dirty worktrees. Do not reset, clean, overwrite, or discard unrelated work.
- Split inherited draft work into focused, reviewable commits before building on it.
- Model downloads and runtime installs must go through the app only.
- Before every app download, obtain two fresh identical plans and require `fitsWithQueue=true` while retaining the 64 GiB reserve.
- Do not delete existing models; they are required for preview-regression testing.
- Generate from a clean, frozen execution checkout. Continue source work in a separate checkout/worktree so generation receipts remain bound to an immutable commit.
- Record the exact source commit, graph hash, template contract hash, artifact revisions, runtime profile/digest, device, dtype, quantization, offload policy, attention backend, seed, and output hash for every run.
- A generated file is not approved merely because the graph completed.
- Images and videos receive Codex quality screening before user review.
- Audio receives automated integrity checks, then goes directly to the user for listening review; Codex must not claim subjective audio approval.
- A failed run remains in the queue until fixed and successfully rerun, or until an explicit evidence-backed external blocker is recorded.

## Current baseline

The last coherent committed coverage ledger is pinned to Diffusers `90b4e34e79a86ec5e7f2437634fe95ecd2108796` and Transformers main `96fe6dce36cc929a5ffd3e34296554c4cb6b669e`.

- Diffusers exports classified: 327
  - executable: 116
  - equivalent: 15
  - contract-only: 20
  - research-blocked: 120
  - intentionally excluded: 56
  - unreviewed: 0
- Finite Transformers semantics: 6
  - executable: 4
  - research-blocked: 2
  - production-supported: 5
- Canonical workflows: 198
- Unique canonical model/mode pairs: 186
- Public templates: 77, covering 51 workflows
- Hidden candidate contracts: 147, covering the workflows without public templates
- Hidden candidates requiring input examples: 92
- Hidden candidates not requiring input examples: 55
- Historically reviewed Gallery examples: 70
- Current exact-complete release receipts: 11
- Resource recipes physically qualified for release: 0/51

Important boundary: `executable` currently proves source/graph admission only. It does not automatically prove a real output, correct quality, hardware portability, license approval, Auto eligibility, Gallery eligibility, or release readiness.

## Current inherited draft state

The backend and client both contain substantial uncommitted work from a later model. Preserve it, but do not treat it as complete.

Known draft areas:

- Shap-E image-to-3D adapter/artifact/spec work exists, but its graph, coverage, candidate contract, client bridge, runtime receipt, and asset are not complete.
- Template authoring drafts exist for 147 hidden candidates.
- Original input-fixture planning exists for 92 input-conditioned candidates.
- A 30-item generic task-gap inventory exists; 28 entries require new generic task boundaries.
- Human-review staging/approval tooling exists but does not consume the new generation-campaign report.
- AudioLDM compatibility code monkeypatches Transformers generation behavior and requires narrow version/model review before acceptance.
- GLM/DreamLite dtype changes conflict with locked tests and require an explicit numerical-contract decision.
- The client draft contains an unsafe E2E optional-runtime bypass that can accept the wrong active runtime profile. Remove it and test exact-profile matching.
- Generated `web/assets` are modified while the client source tree does not pass the full check. Do not commit generated bundles independently.

Current dirty-tree gates observed on 2026-08-20:

- Backend exact-pin suite: 21 failed, 1,994 passed, 4 skipped.
- Client typecheck: passed.
- Client full check: failed formatting on five draft files.

No generation campaign should be called release evidence until these inconsistencies are repaired and the execution checkout is clean.

## Latest upstream delta to admit

Freeze and verify the current official heads again immediately before implementation; upstream can move.

The 2026-08-20 audit found:

- Diffusers main `4e0466f3e5260f0d78b5e2b68ffbf27d819cc6db`, 17 commits beyond the current MoDiff pin.
- Transformers main `94f09cfec149050b5355bab7f207ac69e21f1a02`, 55 commits beyond the reviewed MoDiff main.
- Comfy workflow templates main `0f3d903f4e22bc6d4cf7fc9c5733555f701e4dce`, beyond the current research pin.

Diffusers now exposes 330 static top-level `*Pipeline` classes. The three additions are:

- `StableAudio3Pipeline`
- `StableAudio3AudioToAudioPipeline`
- `StableAudio3InpaintPipeline`

Stable Audio 3 immediate disposition:

- Add all three to the inventory and create explicit contract-only/research records.
- Define generic `text_to_audio`, `audio_to_audio`, and `audio_inpaint` task contracts without adding model-name behavior to the client.
- Do not claim app-downloadable execution yet. Upstream documentation currently uses a locally converted checkpoint and says Diffusers-format checkpoints are not published.
- Record the gated-source and conversion-artifact blocker.
- Enforce float32 on CPU/MPS unless later exact evidence supports another dtype.

Transformers delta immediate work:

- Relock the exact current main archive and deterministic app-owned wheel.
- Review Step 3.7 through the existing generic image/video-to-text semantic rather than inventing a model-specific frontend task.
- Review NVFP4, FlashAttention hub-kernel changes, and audio/video processor fixes for runtime/profile impact.
- Re-run install, activate, workload, rollback, and clean-base qualification on Linux x86-64 before changing the production cutover.

Comfy delta immediate work:

- Regenerate the pinned inventory and authoring-research ledger.
- Classify the new local Wan Animate distilled template.
- Keep hosted/API-only templates ineligible for local execution.
- Do not copy Comfy graphs, custom packages, model files, or media.

## Contract readiness tiers

### Tier A — Structurally ready now

The 198 checked-in canonical workflows have source and graph contracts. They can enter preflight immediately, but each still needs runtime/model/input/resource checks before execution.

The 77 public templates are the safest first regression population because they have established intent and 70 have previously reviewed visual examples.

### Tier B — Fastest new asset candidates

The 55 hidden candidates with no required input examples should be considered first, ordered by:

1. exact graph and authoring hashes current;
2. model snapshot already complete in app cache;
3. required runtime already active and exact;
4. no unresolved model/output rights gate;
5. smallest expected memory/runtime cost;
6. image before short video before long video, while audio is sent directly after integrity checks.

These require authoring finalization, a clean execution receipt, generation, and review—not new generic task boundaries.

### Tier C — Fast after fixture approval

The 92 input-conditioned hidden candidates already have procedural fixture planning. Finalize a small rights-cleared fixture kit and bind each input by hash. The current radio, tram-stop, control-line, and mask fixtures are suitable for technical routing tests, but publication-quality inputs must be explicitly selected and reviewed.

### Tier D — Previously executed on this machine

The last campaign recorded 59 completed workflows:

- 38 image outputs;
- 15 video outputs;
- 6 audio outputs.

This is useful execution evidence, but not approval evidence. After the image-quality cleanup performed on 2026-08-20, 21 image candidates remain, 17 image outputs require fixes and reruns, all 15 videos require Codex temporal/visual review, and all 6 audio files require direct user listening review after integrity checks.

### Tier E — Fast failure recovery

The 107 failed campaign runs are not 107 independent model gaps. The recorded failures include:

- 55 browser/page termination failures;
- 28 graph lifecycle/finalization failures;
- 3 optional-runtime failures;
- 2 missing OpenCV failures;
- 1 incomplete-model failure;
- remaining failures requiring individual classification.

Browser and graph lifecycle failures are the highest-leverage repair because one correct fix can unblock dozens of assets. Fix the campaign runner and graph lifecycle before downloading additional heavy models.

### Tier F — Not yet generation-ready

- Shap-E image-to-3D draft until its source/spec/graph/client/ledger chain is complete and the app can safely install the selected snapshot.
- Stable Audio 3 until an official or reviewed deterministic Diffusers-format artifact path exists.
- Research-blocked/contract-only upstream classes lacking safe immutable artifacts, bounded generic actions, acceptable licenses, or verified component assembly.
- Any workflow whose exact optional runtime/profile is not active.
- Any input-conditioned workflow without a selected rights-cleared input.

## Parallel execution strategy

### Lane 1 — Asset generation and review

Run from a clean, frozen execution checkout while Lane 2 continues in another checkout.

1. Build a machine-readable readiness ledger for all 198 workflows.
2. Select only workflows whose graph, artifacts, inputs, runtime, disk plan, and device recipe pass preflight.
3. Prioritize cached image workflows, then short cached videos. Keep one heavy model generation active at a time on the accelerator.
4. Allow app-only model downloads to run while CPU-only source work proceeds, but never exceed disk reserve and never delete old models.
5. Validate every output before it is considered generated.
6. Route images/videos through Codex screening.
7. Route technically valid audio directly to the user.
8. Store rejected outputs outside the active review shortlist and return their workflow IDs to the rerun queue.

### Lane 2 — Source, runtime, and generic-client work

1. Stabilize inherited changes and restore all gates.
2. Refresh exact upstream pins and ledgers.
3. Move frontend model decisions into backend-published schemas.
4. Resolve upstream source/admission gaps.
5. Regenerate canonical graphs and candidate contracts only after the source catalog is stable.
6. Publish a new execution checkout to Lane 1 only after all source and graph gates pass.

The lanes communicate through immutable revisions and hashes. Lane 1 must never silently consume a dirty or moving Lane 2 checkout.

## Failure policy: fix, prove, rerun

Every run follows this state machine:

`queued -> preflight -> running -> output_integrity -> Codex/user_review -> accepted`

A failure transitions to:

`failed -> classified -> reproduction -> fix -> regression_test -> same-workflow_rerun`

It must not transition from `failed` to `skipped` merely to let the campaign finish.

For every failure:

1. Preserve the exact error, graph, template contract, runtime/profile digest, app state, and relevant logs.
2. Classify it as graph/client lifecycle, backend action, artifact, runtime, memory/resource, dependency, output serialization, or quality failure.
3. Reproduce the smallest failing path.
4. Fix the underlying generic layer when possible; do not add a model-name client special case.
5. Add a focused regression test that failed before the fix.
6. Run the affected broader gates.
7. Rerun the exact workflow with the same locked inputs and seed.
8. Keep it in the retry ledger until a valid output exists or an external blocker is formally recorded.

Quality failure is also a real failure. Adjust prompt/default/resource settings only through reviewed contract changes; bind the new contract hash before rerunning.

## Output integrity and review gates

### All media

- File extension and MIME/file magic must agree.
- Decode must succeed using a second independent decoder where practical.
- Output must match the declared media kind.
- Record dimensions, frame rate, duration, channels/sample rate, and byte size.
- Reject empty, truncated, all-black/all-white, near-flat, NaN/Inf, or implausibly tiny outputs.
- Bind SHA-256 and generation receipt before review.

The current campaign has a concrete packaging bug: all 38 image outputs used `.png` filenames but contained WebP bytes. Fix the exporter before rerunning or publishing.

### Image review by Codex

Check:

- prompt/task adherence;
- source identity and composition preservation for edits;
- mask locality and seam quality for inpaint/outpaint;
- control adherence;
- anatomy, geometry, text, duplication, and artifact defects;
- exposure, sharpness, coherence, and useful resolution;
- comparison with any previously approved example.

Only shortlisted images go to the user. Codex screening is not final user approval or rights approval.

### Video review by Codex

Decode the complete video and inspect contact frames plus temporal playback. Check:

- first/middle/last frames and scene intent;
- identity and geometry continuity;
- flicker, jitter, melting, duplication, and abrupt cuts;
- camera/motion compliance;
- duration, FPS, frame count, resolution, and container integrity;
- source/control timing for conditioned video;
- audio/video sync where applicable.

Rejected videos return to the rerun queue before user review.

### Audio review by the user

Codex performs only technical checks:

- decoding;
- declared duration and sample rate;
- channel count;
- clipping/near-silence/DC-offset checks;
- abrupt truncation detection;
- expected file/container type.

Technically valid audio is then sent directly to the user for subjective review.

## Older public templates: confidence and verification without full inference

Current confidence is mixed:

- Visual intent confidence is high for the 70 previously reviewed Gallery examples.
- Current-release evidence confidence is high for only 11 templates with exact current receipts.
- Forty-four historical examples are stale relative to current graph/runtime evidence.
- Seven have no historical execution evidence; one of these is a user-supplied-artifact exemption.

Do not infer that old templates are correct merely because their labels and screenshots remain present. Also do not blanket-rerun all 77.

Use the following verification ladder, cheapest first:

### Level 1 — Exact contract reconciliation

Compare the old approved receipt with the current template and graph:

- canonical workflow ID;
- semantic graph hash, ignoring node positions and other visual-only metadata;
- model repo/revision and selected-file hashes;
- prompt, negative prompt, seed, scheduler, steps, guidance, dimensions, and mode-specific fields;
- adapter/component revisions;
- generic node/action IDs and field bindings;
- output contract;
- runtime profile and dependency versions.

If everything execution-relevant is identical, preserve the old visual approval. No inference is required.

### Level 2 — Static graph and schema replay

Load every canonical graph through the current backend/client parsers without model weights and prove:

- all nodes/actions exist;
- handles, types, required inputs, and edges resolve;
- no isolated/unreachable managed nodes exist;
- current backend field schemas accept every stored value;
- generic node roles map to the same backend adapter/action;
- artifact and optional-runtime requirements resolve exactly;
- graph serialization round-trips without semantic changes.

This is the best inexpensive proof that migration to generic nodes did not structurally break older templates.

### Level 3 — Fake-pipeline behavioral contracts

Use signature-compatible fake pipelines and synthetic tensors/media to execute each generic action path. Assert:

- exact parameter forwarding and omission of unsupported fields;
- source/control/mask routing;
- bounds and cancellation;
- callback and progress behavior;
- output shape/type/serialization;
- inpaint/outpaint compositing locality;
- runtime/profile mismatch rejection.

This checks backend semantics without loading model weights or running denoising.

### Level 4 — Component/load canaries

For templates affected by dependency or loader changes, instantiate the exact local pipeline and components with `local_files_only` and safe serialization, but do not run the full workflow. Verify scheduler, tokenizer/processor, component dtype/device, adapters, and optimization application.

### Level 5 — Tiny targeted numerical canaries

Run one very small, bounded inference only for families whose numerics may have changed—for example GLM/T5 dtype, scheduler behavior, quantization, attention backend, or adapter fuse/hotswap. Use 64–256 px or a very short clip/audio segment as appropriate. Compare deterministic hashes where possible; otherwise compare latent statistics, prompt embeddings, perceptual hashes, SSIM/LPIPS/CLIP similarity, and safety/integrity invariants.

### Level 6 — Full output rerun

Required only when:

- Levels 1–5 cannot reconcile the current behavior;
- an execution-relevant graph/model/runtime field changed;
- a targeted canary fails;
- the existing asset is missing or corrupt;
- current release policy requires a new physical resource receipt.

This ladder confirms generic-node migrations cheaply while limiting expensive full generations to affected families.

## Prioritized work plan

### P0 — Make generation reliable immediately

Acceptance target: a frozen clean checkout can run a bounded campaign without browser death, graph lifecycle races, wrong-runtime bypasses, MIME mismatches, or silent skips.

1. Snapshot and split inherited backend/client draft changes.
2. Remove the E2E optional-runtime bypass and require the exact profile ID and digest.
3. Fix browser/page lifecycle and graph-finalization races with focused regressions.
4. Fix WebP-as-PNG output naming/serialization.
5. Install/route the OpenCV dependency through the correct app/runtime profile rather than direct ad hoc installation.
6. Reconcile the one incomplete model through two app plans and space policy; do not delete older models.
7. Build a retry ledger for all 107 failures.
8. Rerun the 17 removed image workflows after their technical or quality issue is corrected.
9. Review all 15 existing videos before presenting them to the user.
10. Perform integrity checks on the six audio files and present them directly to the user.

Estimated active agent time: 6–12 hours, excluding model inference/download duration.

### P1 — Lock the old-template compatibility proof

Acceptance target: all 77 public templates have a Level 1–3 compatibility result, and only materially affected families are selected for canaries/full reruns.

1. Generate an exact old-vs-current reconciliation ledger.
2. Run static graph/schema replay for all 198 workflows.
3. Run fake-pipeline behavioral tests for every generic action topology.
4. Classify each public template as exact-compatible, canary-required, or full-rerun-required.
5. Preserve the 70 old quality approvals unless a real execution-relevant change is proven.

Estimated active agent time: 4–8 hours.

### P2 — Refresh latest upstream pins and contracts

Acceptance target: current heads are frozen, every latest upstream class/semantic is classified, and generated ledgers are exact.

1. Recheck official heads.
2. Relock Diffusers and Transformers artifacts/source.
3. Add the three Stable Audio 3 records and generic task contracts.
4. Reconcile Step 3.7 and runtime/quantization deltas.
5. Refresh Comfy catalog/research ledgers.
6. Regenerate coverage and provenance tests.

Estimated active agent time: 4–8 hours. Stable Audio 3 live execution remains artifact/access dependent.

### P3 — Complete model-neutral client architecture

Acceptance target: adding a backend model/spec requires no model-name conditional in the client.

1. Extend backend capability schemas with complete UI, graph, artifact, and optimization metadata.
2. Add strict schema validation and generated/shared types.
3. Replace handwritten static client profiles with authoritative backend records.
4. Replace model-name template/runtime branches with contract fields.
5. Replace model-specific UI controls with schema-rendered generic fields.
6. Retain a narrowly scoped migration map for old saved workflows only.
7. Add a source test that rejects new model-name behavior branches outside approved migration/catalog files.

Estimated active agent time: 12–24 hours.

### P4 — Close fast source/admission gaps

Acceptance target: every safely reusable existing artifact/class is either admitted or explicitly blocked with evidence.

1. Finish or explicitly defer Shap-E image-to-3D.
2. Repair/reject the AudioLDM compatibility monkeypatch.
3. Resolve GLM/DreamLite dtype changes through exact tests/canaries.
4. Implement the highest-value generic task gaps that reuse existing artifacts.
5. Reclassify every affected Diffusers/Transformers entry and regenerate graphs/contracts.

Estimated active agent time: 1–3 continuous agent-days depending on artifact/security findings.

### P5 — Produce the asset review queues

Acceptance target: every generation-ready contract has a valid asset or a precise unresolved external blocker.

1. Generate the 55 input-free hidden candidates first.
2. Finalize rights-cleared inputs for the 92 conditioned candidates.
3. Continue cached-model batches while P2–P4 source work proceeds.
4. Codex-screen every image/video.
5. Send technically valid audio directly to the user.
6. Create small, navigable review batches grouped by media and model family.
7. Apply user approval only through explicit approval tooling.

Estimated agent orchestration/review time: 6–12 hours plus model inference time. Heavy generation duration depends on the accelerator, model cache, and workflow length.

### P6 — Platform and optimization qualification

Acceptance target: each published workflow has an explicit supported platform/device/resource recipe and tested fallback ladder.

Required platform rows:

- Linux x86-64 CPU
- Linux x86-64 NVIDIA CUDA
- Linux x86-64 AMD ROCm
- Windows x86-64 CPU/CUDA
- macOS ARM64 CPU/MPS
- Linux ARM64 and Windows ARM64 only where dependencies genuinely support them

Per-family policy must choose among:

- CUDA FlashAttention 2/3 or xFormers when supported;
- native PyTorch SDPA/efficient attention fallback;
- MPS/CPU math fallback;
- model-supported bitsandbytes, TorchAO, Quanto, GGUF, or native quantization only where qualified;
- resident, model CPU offload, sequential offload, or group offload only where compatible;
- explicit prohibition of unsafe quantization/offload combinations.

Fill all 51 resource recipes with real measurements: load/run time, peak host memory, peak device memory, output integrity, runtime profile, restart/rollback behavior, and exact receipt hashes.

Estimated agent work: 1–2 days of orchestration and evidence processing. Wall-clock completion depends on access to the physical target machines and model inference duration.

## Current 21-image Codex shortlist

These images passed the first Codex visual/task screen. They are not yet user-approved or rights-approved.

1. [AuraFlow — text to image](/home/sayak/MoDiff/MoDiff/review-pending/AuraFlowPipeline__text_to_image/campaign-auraflowpipeline__text_to_image-gpu-v1.png)
2. [Chroma — text to image](/home/sayak/MoDiff/MoDiff/review-pending/ChromaPipeline__text_to_image/campaign-chromapipeline__text_to_image-gpu-v1.png)
3. [CogView3 Plus — text to image](/home/sayak/MoDiff/MoDiff/review-pending/CogView3PlusPipeline__text_to_image/campaign-cogview3pluspipeline__text_to_image-gpu-v1.png)
4. [DreamLite — text to image](/home/sayak/MoDiff/MoDiff/review-pending/DreamLitePipeline__text_to_image/campaign-dreamlitepipeline__text_to_image-gpu-v1.png)
5. [Ernie Image — text to image](/home/sayak/MoDiff/MoDiff/review-pending/ErnieImagePipeline__text_to_image/campaign-ernieimagepipeline__text_to_image-gpu-v1.png)
6. [FLUX.2 Klein — text to image](/home/sayak/MoDiff/MoDiff/review-pending/Flux2KleinPipeline__text_to_image/campaign-flux2kleinpipeline__text_to_image-gpu-v1.png)
7. [FLUX Krea — text to image](/home/sayak/MoDiff/MoDiff/review-pending/FluxKreaPipeline__text_to_image/campaign-fluxkreapipeline__text_to_image-gpu-v1.png)
8. [HunyuanDiT — text to image](/home/sayak/MoDiff/MoDiff/review-pending/HunyuanDiTPipeline__text_to_image/campaign-hunyuanditpipeline__text_to_image-gpu-v1.png)
9. [Kandinsky 3 — text to image](/home/sayak/MoDiff/MoDiff/review-pending/Kandinsky3Pipeline__text_to_image/campaign-kandinsky3pipeline__text_to_image-gpu-v1.png)
10. [LongCat Image — text to image](/home/sayak/MoDiff/MoDiff/review-pending/LongCatImagePipeline__text_to_image/campaign-longcatimagepipeline__text_to_image-gpu-v1.png)
11. [Lumina 2 — text to image](/home/sayak/MoDiff/MoDiff/review-pending/Lumina2Pipeline__text_to_image/campaign-lumina2pipeline__text_to_image-gpu-v1.png)
12. [Marigold — depth estimation](/home/sayak/MoDiff/MoDiff/review-pending/MarigoldDepthPipeline__depth_estimation/campaign-marigolddepthpipeline__depth_estimation-gpu-v1.png)
13. [PRX — text to image](/home/sayak/MoDiff/MoDiff/review-pending/PRXPipeline__text_to_image/campaign-prxpipeline__text_to_image-gpu-v1.png)
14. [Qwen Image Modular — text to image](/home/sayak/MoDiff/MoDiff/review-pending/QwenImageModularPipeline__text_to_image/campaign-qwenimagemodularpipeline__text_to_image-gpu-v1.png)
15. [Sana — text to image](/home/sayak/MoDiff/MoDiff/review-pending/SanaPipeline__text_to_image/campaign-sanapipeline__text_to_image-gpu-v1.png)
16. [Sana Sprint — text to image](/home/sayak/MoDiff/MoDiff/review-pending/SanaSprintPipeline__text_to_image/campaign-sanasprintpipeline__text_to_image-gpu-v1.png)
17. [Stable Diffusion PAG — text to image](/home/sayak/MoDiff/MoDiff/review-pending/StableDiffusionPAGPipeline__text_to_image/campaign-stablediffusionpagpipeline__text_to_image-gpu-v1.png)
18. [Stable Diffusion XL PAG — text to image](/home/sayak/MoDiff/MoDiff/review-pending/StableDiffusionXLPAGPipeline__text_to_image/campaign-stablediffusionxlpagpipeline__text_to_image-gpu-v1.png)
19. [Stable Diffusion XL — text to image](/home/sayak/MoDiff/MoDiff/review-pending/StableDiffusionXLPipeline__text_to_image/campaign-stablediffusionxlpipeline__text_to_image-gpu-v1.png)
20. [Stable Diffusion XL Turbo — text to image](/home/sayak/MoDiff/MoDiff/review-pending/StableDiffusionXLTurboPipeline__text_to_image/campaign-stablediffusionxlturbopipeline__text_to_image-gpu-v1.png)
21. [Z-Image Modular — text to image](/home/sayak/MoDiff/MoDiff/review-pending/ZImageModularPipeline__text_to_image/campaign-zimagemodularpipeline__text_to_image-gpu-v1.png)

All 21 files currently contain WebP bytes despite their `.png` suffix. Correct the format/extension during review-package preparation; do not silently alter the generation receipt without recording the new byte hash.

## Removed image outputs

On 2026-08-20, the 12 rejected and 5 borderline image outputs were removed from their original campaign paths using `gio trash`. They are recoverable from the desktop trash until it is emptied. Their workflow IDs remain in the rerun queue and must not be marked skipped or complete solely because the old file is absent.

## Definition of done

The campaign is complete only when:

- the client contains no model-specific execution/UI decisions outside reviewed catalog/migration boundaries;
- current exact Diffusers/Transformers/Comfy revisions are pinned and fully classified;
- every safely supported task has generic nodes, exact backend adapters, canonical workflows, and template/authoring contracts;
- every failed generation has been fixed and rerun or has a precise external blocker;
- all intended images/videos have passed Codex screening before user review;
- all intended audio has passed integrity checks and been sent to the user;
- user approvals and rights approvals are explicitly recorded;
- every published workflow has a physically measured supported resource recipe and fallback policy for its claimed platforms;
- all backend/client/source/graph/release gates pass from clean checkouts;
- all local commits intended for delivery are pushed to the correct remote branches.

## Initial instruction for the next chat

Continue this plan persistently. Begin with P0 and maintain the two parallel lanes. Preserve the inherited dirty worktrees and the 21-image shortlist. Do not skip failed workflows. Do not ask for image/video review until Codex has screened them; send technically valid audio directly for user listening review. Use app-only downloads with two fresh matching space plans, preserve all existing models, and stop only for a real external blocker or required user intervention.
