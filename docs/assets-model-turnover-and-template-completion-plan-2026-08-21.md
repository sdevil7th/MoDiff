# MoDiff asset, model-turnover, and template-completion execution plan

Status: active execution plan  
Created: 2026-08-21  
Owner: MoDiff engineering and qualification campaign  
Scope: backend, client, frozen generation checkout, public-template migration, hidden/new templates, review assets, cached-model turnover, and subsequent downloads

This document is the current operational plan. It supplements
`docs/generic-nodes-upstream-assets-completion-plan.md` and supersedes that document wherever the two conflict about asset quality, retry policy, cached-model eviction, or execution order. The older document remains useful as the source-coverage and generic-node architecture record.

## 1. Required outcome

Finish the template and asset program without weakening the generic-node architecture:

1. A user can click a running/completed run notification and reach the exact owning workflow without an exception.
2. The 77 existing public templates remain loadable after migration, with bounded live canaries only where historical evidence is stale.
3. New template candidates are qualified with researched prompts, parameters, inputs, and real outputs—not merely structurally valid graphs.
4. Models already present in the Hugging Face cache are exhausted first: finish every in-scope workflow for a repository, preserve accepted evidence, then remove the exact revision through the app.
5. Reclaimed disk is used for the next planned model batch; downloads never bypass the app planner, the 64 GiB reserve, licenses, or exact revision checks.
6. Every open-source model, Diffusers pipeline/action, Transformers task, optional runtime, and quantization path exposed by the app has user-facing acquisition and usage documentation.

No model-specific nodes may be added to make a template pass. Nodes stay generic and model-neutral. Exact model behavior belongs in backend-discovered field definitions, capability/profile records, template data, and immutable execution receipts.

## 2. Non-negotiable execution gates

### Gate A — notification navigation before generation

No new template asset generation may start until all of these pass:

- run-activity unit and lifecycle regressions;
- mocked notification/shelf navigation in Playwright;
- a live-backend test using a real recent run record and its real workflow snapshot;
- no `QuotaExceededError`, unhandled rejection, wrong-tab navigation, or lost task identity.

Current state: **implemented and green on 2026-08-21**.

The reproduced failure was concrete: `/workflows` returned 946 backend-owned documents totaling about 36.2 MB. The client treated every document as an open browser tab and Zustand attempted to copy them all into `localStorage`. Browser quota exhaustion then broke workflow switching and caused the backend-sync effect to retry every two seconds.

Implemented correction:

- backend workflows remain a saved-document library;
- startup hydrates only workflow IDs that are already open in that browser;
- unopened websocket workflow updates do not create tabs;
- the local crash checkpoint is bounded to 12 tabs and always retains the active tab;
- a failed local checkpoint cannot throw through normal UI actions;
- run restoration has an error boundary that opens the exact Queue task when a snapshot cannot be restored.

Evidence:

- 59 focused run/lifecycle tests passed;
- TypeScript passed;
- live Playwright opened task `QWdsR8VlMCPQ`, restored workflow `QRuV_2qFbhaDOj0iLINGn`, selected Studio, emitted no page error, and kept the local checkpoint below 2 MB; the repeatable focused test passed in 36.6 seconds.

### Gate B — research before a generation attempt

Every model-quality attempt needs a small research dossier recorded with the campaign recipe:

- official model card and official Diffusers/Transformers documentation or source;
- exact base model, adapters, scheduler, revision, dtype, and runtime profile;
- recommended resolution, frame/sample count, steps, guidance, negative prompt, and conditioning rules;
- a prompt/input designed to expose the advertised capability;
- measurable acceptance criteria for the media kind.

No identical retry is allowed. A second attempt must cite a materially different hypothesis. After two quality failures, stop until there is an engineering change, a different official recipe, or an explicit reviewed exception.

### Gate C — quality before approval

Technical success is not Gallery approval.

- Image: useful card resolution, coherent subject/anatomy, prompt fidelity, no obvious broken objects, no accidental theme repetition.
- Video: real temporal change, legible motion, no near-static slideshow, no severe blur/flicker, correct duration/FPS, and a thumbnail/preview that reads as motion.
- Audio: normally at least 30 seconds for music showcases, clear intended content, no unexplained single tone, discontinuity, clipping, or synthetic placeholder input.
- Transform/utility: show a meaningful before/after pair or comparison; an isolated output that hides the operation is technical proof only.
- 3D: a viewable showcase size and orbit/motion sufficient to inspect the object.

Automated screens can reject an asset but cannot grant final visual/listening approval.

### Gate D — deletion and download safety

No cached repository may be removed until:

- all canonical workflow dependencies for that exact repository/revision are enumerated;
- all public migration canaries that use it are complete;
- every intended workflow has a durable accepted asset or a recorded non-executable blocker;
- technical metrics, output hashes, runtime fingerprint, and exact model revision are preserved;
- quality and rights/provenance states are explicit;
- no run or download is active;
- the app shows the exact revision and affected workflows before confirmation.

No new download may start without two fresh, identical app-generated download plans with `fitsWithQueue=true` and at least 64 GiB reserved after queued bytes.

## 3. Verified baseline

### 3.1 Canonical workflow and migration structure

- 199 canonical workflows exist in `data/workflow-library-manifest.json`.
- All 199 graph documents round-trip, hash, parse, and pass the current structural catalog checks.
- The focused generic fake-pipeline/action matrix passed 591 tests with 23 platform/live-only skips.
- Client template tests passed: 129 template, 16 task-template, 5 quality tests; workflow verification reported all 199 supported.
- All 77 public templates resolve to current canonical workflows.
- A legacy persisted template tab is adopted as a managed tab without moving or replacing its saved graph.

These results prove structural migration and generic action routing. They do not prove that all 77 have fresh live outputs.

### 3.2 Public-template evidence remaining

The authoritative reconciliation is
`review-pending/public-templates/reconciliation.v1.json`:

| Bucket | Count | Required work |
| --- | ---: | --- |
| `current_keep` | 26 | Preserve historical bytes; do not blanket-rerun. Close missing physical metrics separately if needed. |
| `stale_canary_required` | 44 | Run one targeted live canary on the exact current graph/runtime when its model remains cached. |
| `never_had_example` | 7 | Author, generate, review, and publish the first example. |

The 44 stale records break down into 24 graph/node mismatches, 10 graph/node mismatches plus retired historical node names, 6 template-lock mismatches, 1 lock plus graph mismatch, and 3 lock plus prompt plus graph mismatch. Retired Qwen/LTX names are a migration boundary; they must not be restored as model-specific nodes.

The seven first-example public templates are:

1. `ace_step_chinese_new_year_lora`
2. `ace_step_custom_lora`
3. `flux_lora_cinematic_octane_3d`
4. `wan_21_t2v_13b_seed_vault`
5. `wan_22_i2v_seed_vault`
6. `wan_vace_masked_object_replace`
7. `wan_video_long_showcase`

The 26 `current_keep` IDs are retained in the reconciliation ledger and should be treated as immutable historical examples unless a receipt-specific deficiency requires a metadata-only repair.

### 3.3 Release-evidence blocker

`npm run release:contract:verify` currently reports a stale release contract because this host lacks the 70 pinned `*.reviewed-provenance.json` files from the app-owned Gallery dataset. Running `release:contract:generate` in that state would incorrectly erase retained historical evidence. The safe choices are:

1. install and verify the hash-pinned Gallery dataset, then verify/regenerate; or
2. change check mode so retained, hash-pinned release evidence remains authoritative when the dataset is absent.

Do not regenerate the release contract until one of those is implemented.

### 3.4 Current review state

The human feedback record is `review-pending/user-quality-review-2026-08-21.v1.json`; the quality incident and corrective policy are in `docs/review-campaign-quality-incident-2026-08-21.md`.

Before the rejected-asset cleanup, the staged review index deduplicated to 38 items:

- 10 quality-approved but rights/provenance pending;
- 11 rejected;
- 17 awaiting human review.

The feedback record contains 48 decisions because it also classifies duplicate/alternative campaign files and operation-only proofs. Approved bytes must be preserved. Rejected assets must not be silently replaced or promoted. Near-static videos are rejected as a class, not one by one after wasting another review cycle.

There is a receipt/index consistency defect to fix before turnover. AuraFlow, ERNIE, LongCat Image, and PRX media files exist in the main `review-pending` tree and the user feedback ledger records quality approval, but the current main review index still lists those workflows under `blockedGeneration` as not generated. The frozen generation checkout has quality-acceptance JSON only for AuraFlow and PRX, and both still contain `userApproval: pending`; ERNIE and LongCat Image currently have media but no matching acceptance JSON beside it. The reconciliation step must create or update exact immutable receipts from the task/run evidence and user decision, then rebuild the index. It must not infer rights approval.

Cleanup completed on 2026-08-21: 55 rejected/withdrawn files (23,316,849 bytes) were moved out of `review-pending` to the recoverable Trash quarantine at `/home/sayak/.local/share/Trash/files/modiff-review-rejected-2026-08-21`. Explicit approvals and the SanaVideo, SDXL-PAG, Wan VACE, and Wan Video “improve but keep” fallbacks were preserved. There are 25 top-level approved/fallback media files and no rejected media left under `review-pending`.

The first exact campaign reconciliation is also complete for AuraFlow, ERNIE, LongCat Image, and PRX. Each main approved file was matched byte-for-byte to persisted `data/studio/outputs.json` evidence, including the original task ID, graph snapshot, repository, and immutable 40-character model revision. Per-asset `campaign_generation_review_receipt` files now preserve that evidence and the user quality decision. After the two upscaling candidates described below were added, the review index has 21 valid visible cards, zero broken asset references, 14 quality-approved/right-pending cards, 7 awaiting review, and 23 `removedReviewItems` audit records. The approved campaign receipts correctly remain turnover-ineligible where rights or runtime evidence is pending. None of those gaps was guessed or papered over.

### 3.5 First new-template quality milestone

Two app-managed Real-ESRGAN x2 workflows now have exact live ROCm evidence and are classified `execution-qualified-gallery-review-pending`:

- `SpandrelImageUpscale:image_upscale`: three generic nodes (`Image.Load` → `Spandrel.Upscaler` → `Image.Preview`), 512×256 JPEG input, 1024×512 WebP output, 13.94 seconds, 1.22× edge-detail energy versus bicubic, and 0.021 reconstruction MAE.
- `SpandrelVideoUpscale:video_upscale`: two generic nodes (`Video.UpscaleVideo` → `Video.Export`), 81-frame 16 fps 512×288 input, 81-frame 16 fps 1024×576 output, 5.06 seconds preserved, 62.70 seconds execution, 1.19× detail energy versus bicubic, and 0.018 reconstruction MAE.

Both candidates include exact model/revision/file hashes, source-fixture provenance, runtime fingerprints, task/run hashes, automated reports, manual Codex shortlist screens, and matched before/after cards. Automated screens are rejection-only. Both remain `pending_human_review`, with user approval, rights approval, Gallery registration, dataset publication, and model-turnover eligibility all false.

The video run was not accepted merely because it produced MP4 bytes. Its source was replaced before generation: the earlier one-second/eight-frame synthetic canary measured almost no motion and was excluded. The selected five-second greenhouse push-in has approximately 18% materially changing pixels per transition. The first bounded run then exposed one implementation defect—the node returned an undeclared `fps` output key. The output schema was corrected, 21 focused contract tests passed, and one code-fix-driven retry succeeded. No identical recipe retry occurred.

### 3.6 Cached hidden-template checkpoint — 2026-08-22

The next cached, repository-complete cohort is being worked through with the same rejection-first policy:

- `JoyImageEditPipeline:text_to_image` is now shortlisted for workspace-owner review. The live graph used only generic quantization, recipe, pipeline-loader, image-generate, and preview roles. Selecting JoyAI dynamically bound its exact repository and the official 40-step, 1024×1024, guidance-4, 4096-token recipe. One pre-denoising canary exposed a stale model-scoped 2048-token adapter bound; current Diffusers 0.40 and its official API both declare 4096. The bound was corrected with focused regression coverage, then one media-producing run passed every objective screen and native/card inspection. User approval, rights approval, publication, and deletion eligibility remain pending.
- `JoyImageEditPipeline:edit_image` used the shortlisted 1024px violin-workshop image as a provenance-bound source and the model card's exact camera-control prompt pattern. Its first native attempt produced an all-black image and was rejected before review. Component probes located an edit-only ROCm numerical boundary: the source/VAE path stayed finite, but Qwen3-VL's image-conditioned prompt embeddings became entirely non-finite in BF16. A generic mode-scoped adapter bridge now runs only the JoyAI edit encoder in FP32, checks the multimodal embeddings, and casts them to the BF16 transformer boundary; the text-to-image path remains unchanged. Focused regressions and a real 512px one-step canary passed. The single code-fix-driven full rerun then produced a sharp, coherent camera change of the same scene. Its initial raw-correlation screen exposed a separate quality-gate defect: same-pixel correlation cannot validate a deliberate viewpoint transformation. Following OpenCV's feature-matching/homography guidance, the gate now requires RANSAC-supported ORB evidence, material overlap, and aligned correlation as an alternative retention proof; unrelated/noisy edits still fail, and 12 focused tests pass. The output passed all automated checks with 61 inliers, 75% overlap, and 0.436 aligned correlation, but native inspection rejected it because the closer framing cuts off the violin scroll. Bounded attempts are exhausted; no third JoyAI edit run is allowed without a new decision and materially different reviewed strategy. The repository remains non-evictable while the text-to-image shortlist awaits user/rights decisions.
- `OmniGenPipeline` text-to-image, edit, and multi-reference modes each reached real execution, but the bounded quality attempts failed native prompt/anatomy/reference-fidelity inspection. All rejected media was moved to recoverable Trash and the family is `quality_blocked`; do not retry without a different official recipe or an engineering change.
- `HunyuanDiTPAGPipeline:text_to_image` exposed two concrete contract defects: the adapter targeted PAG block 1 instead of the official block 14, and its long prompt exceeded the CLIP 77-token surface. Both were corrected and tested. The corrected output improved materially but still failed native geometry/text/path fidelity, so it was rejected and the workflow is `quality_blocked` rather than retried again.
- `flux_lora_cinematic_octane_3d`, one of the seven public templates that never had an example, now has two exact live generic-graph attempts and remains `quality_blocked`. The first official 24-step, guidance-3, 768x1024 run was objectively soft and turned ambiguous scar language into broad facial trauma. After official BFL/LoRA research, only the canonical prompt changed: complete framing, sharp eyes, healed-scar/no-trauma wording, and concrete skin/fabric/metal microtexture; both triggers, adapter weights, seed, dimensions, steps, and guidance stayed fixed. All 199 workflow hashes and all 129 template tests passed before the one corrected run. That run completed in one task but still failed the objective detail floor (0.004557 versus 0.008), produced a fresh bloody cheek wound, and omitted the brass pressure collar. Both rejected assets were quarantined; no third attempt is permitted without a newly reviewed concept/seed and a concise prompt validated in a cheap canary first. Generic nodes were not specialized.
- `PixArtSigmaPipeline:text_to_image` is now shortlisted for workspace-owner review after a numeric qualification and one full generation. The earlier 512px/16-step/FP16 campaign output was black. A two-step component probe showed all 65,536 PixArt latent elements become non-finite in BF16 on this ROCm APU, while a coherent FP32 path remained finite (standard deviation 0.311883); a mixed FP32-transformer/BF16-text path was invalid because PixArt derives the latent dtype from the prompt embeddings. The generic PixArt standard and PAG profiles therefore use coherent FP32 on this qualified host, with no model-named execution node. The canonical generic graph dynamically binds native 1024x1024, 20 steps, guidance 4.5, and 300 T5 tokens. The single authorized full run produced a 1024px kinetic-sculpture image that passed every objective check (edge energy 0.042932, luminance standard deviation 0.271577, perceptual-neighbor distance 19) and native/card inspection. Its exact task, run, repository revision, output hash, research dossier, quality report, and shortlist receipt are staged under `review-pending/PixArtSigmaPipeline__text_to_image`. It is not self-approved: user quality review, rights review, publication, and model turnover remain pending. The repository must also be retained until the PAG sibling has a final disposition.
- `PixArtSigmaPAGPipeline:text_to_image` was researched and corrected before its only full attempt. Official Diffusers uses `blocks.14`, PAG scale 4, and guidance 1 for PixArt PAG; MoDiff had implicitly inherited `blocks.1` and used PAG 3/guidance 4.5. The generic adapter, backend capability, client dynamic profile, campaign recipe, canonical graph, hash-pinned ledgers, and regression tests now share the official contract. The exact 1024px FP32 output passed the rejection-only technical gate (edge energy 0.047379; perceptual-neighbor distance 24) but failed native inspection decisively: it rendered one globe-like sphere inside one pseudo-lettered ring, omitting the required central sun, three planets, concentric arms, and gear train. It was rejected before human review, the media/card were quarantined, and no unchanged or seed-only retry is permitted. The workflow is `quality_blocked`; its sibling standard output remains the sole PixArt shortlist awaiting user/rights review, so the shared repository is still not turnover-eligible.
- `LongCatImageEditPipeline:edit_image` is now shortlisted for workspace-owner review on its first model attempt. Its prior campaign record failed at the graph-finalization boundary before loading a model or generating media. Official LongCat research established the exact BF16, model-offload, 50-step, guidance-4.5, seed-43 recipe and identifies precise editing plus non-edited-region consistency as the core capability. The already user-quality-approved LongCat tram-stop image was bound as an exact-hash internal source, with source rights still pending. A localized instruction changed only the red umbrella canopy to mustard-yellow waxed canvas. The 1024px output passed native/card inspection and retained person, pose, clothing, umbrella geometry/shaft/handle, shelter, bench, timetable, rails, buildings, lighting, reflections, perspective, and framing. The initial global-MAE gate falsely classified this precise edit as unchanged, so the rejection-only evaluator was corrected—not the media rerun—to admit a high-delta localized region only when strong composition evidence is also present. Twenty-three quality/review tests pass; the final evidence records 5.1579% localized high-delta pixels, 0.925 raw and 0.969 aligned correlation, 1,805 inliers, 97.6% inlier ratio, and 99.6% overlap. The exact task/run/model/source/output hashes and before/after card are staged under `review-pending/LongCatImageEditPipeline__edit_image`. User quality approval, source/model/output rights approval, publication, and turnover remain pending. Because this exact 29.31 GB edit repository has one current workflow, approval plus rights closure would make it a high-value app-deletion candidate.
- `StableDiffusionXLInstructPix2PixPipeline:edit_image` was researched against the official 768px FP16, 30-step, guidance-3.0, image-guidance-1.5 recipe and the exact cached 12.63 GB revision. Its earlier browser-closed campaign had never reached inference. The first corrected campaign exposed a real generic-binding defect before model load: the execution spec also routed `conditioningScale=1.5` into the unrelated 0–1 `reference_strength` field. The generic spec now removes that binding for InstructPix2Pix, retains only `image_guidance_scale`, has 44 focused backend tests, and regenerated/verified the canonical graph with no hidden reference-strength value. The first and only model attempt then completed in 30.50 seconds, but both gates rejected it. The output was coherent blue botanical line art on white paper rather than a recognizable Prussian-blue cyanotype, several flower forms were redrawn, aligned overlap was zero with only four inliers, and the comparison-card retention check failed. The media/card were moved to recoverable Trash, the rejection receipt remains under `review-pending/rejections`, and the repository stays `quality_blocked`; no unchanged or seed-only retry is allowed.

- `CogView4Pipeline:text_to_image` is now quality-blocked after a researched corrective attempt. Native review first rejected the frozen 1024px/28-step green-kettle candidate as soft, geometrically weak, and too trivial for a 6B-model showcase; it was never copied into the main review queue. Official CogView4 sources require BF16/FP32, dimensions divisible by 32, at most 2^21 pixels, and demonstrate 50 steps at guidance 3.5. The replacement used a new 1280×768 observatory-library brief and seed. An initial 1280×720 submission exposed a contradiction in the official card, which lists that size in its memory table while also requiring multiples of 32; MoDiff rejected height 720 before denoising with zero accelerator allocation. The corrected single denoising run completed in 327.26 seconds and passed objective size/detail/duplicate gates, but native inspection rejected it: two windows instead of one, no spiral stair, repetitive bars instead of a book wall, and synthetic render quality. The media/card were moved to recoverable Trash. Research and two rejection receipts remain; no third attempt is allowed. The campaign runner now reads reviewed research dossiers and excludes every `quality_blocked*` workflow until its dossier records a materially new resolution, preventing broad campaigns from silently looping rejected families. Seventy-one runner tests pass.

This checkpoint does not authorize model eviction. The JoyAI repository remains required until its edit side has a final disposition and the workspace owner has made the recorded rights/turnover decision.

## 4. What “new templates” means

There are two distinct new-work queues.

### 4.1 Seven public templates with no example

These already exist as public template definitions but have never had a Gallery asset. They are the highest-priority “new public example” work, subject to cached dependencies and the research gate.

### 4.2 148 hidden candidate contracts

`data/template-candidate-contracts.v1.json` and
`data/template-authoring-specs.v1.json` contain 148 hidden candidates complementing the public set:

- 100 image;
- 33 video;
- 6 audio;
- 9 JSON/data;
- 55 input-free;
- 93 requiring selected source media, masks, controls, references, or rights-cleared fixtures.

All 148 have canonical defaults and draft authoring specs. The two upscaling candidates now have execution-qualified, Gallery-review-pending evidence; the remaining candidates stay generation-pending. None becomes public merely because its graph is valid.

The 55 input-free candidates are:

```text
AllegroPipeline:text_to_video
AnimateDiffPAGPipeline:text_to_video
AnimateDiffPipeline:text_to_video
AnimateLCMPipeline:text_to_video
AudioLDM2Pipeline:text_to_audio
AuraFlowPipeline:text_to_image
BuiltinDataOperation:data_conversion
BuiltinDataOperation:graph_utility
BuiltinDataOperation:text_select
ChromaPipeline:text_to_image
CogVideoXPipeline:text_to_video
CogView3PlusPipeline:text_to_image
CogView4Pipeline:text_to_image
ConsistencyModelPipeline:unconditional_image
DDIMPipeline:unconditional_image
DDPMPipeline:unconditional_image
DreamLiteMobilePipeline:text_to_image
DreamLitePipeline:text_to_image
ErnieImagePipeline:text_to_image
GlmImagePipeline:text_to_image
HuggingFaceAnyToAnyModel:text_generation
HuggingFaceAnyToAnyModel:text_to_image
HuggingFaceTextGenerationModel:text_generation
HunyuanDiTPAGPipeline:text_to_image
HunyuanDiTPipeline:text_to_image
JoyImageEditPipeline:text_to_image
Kandinsky3Pipeline:text_to_image
LTX2ConditionPipeline:text_to_video
LTX2Pipeline:text_to_video
LatentConsistencyModelPipeline:text_to_image
LattePipeline:text_to_video
LongCatAudioDiTPipeline:text_to_audio
LongCatImagePipeline:text_to_image
Lumina2Pipeline:text_to_image
LuminaPipeline:text_to_image
MochiPipeline:text_to_video
NucleusMoEImagePipeline:text_to_image
OmniGenPipeline:text_to_image
OvisImagePipeline:text_to_image
PRXPipeline:text_to_image
PixArtSigmaPAGPipeline:text_to_image
PixArtSigmaPipeline:text_to_image
SanaPAGPipeline:text_to_image
SanaPipeline:text_to_image
SanaSprintPipeline:text_to_image
SanaVideoPipeline:text_to_video
ShapEPipeline:text_to_3d
StableAudioPipeline:text_to_audio
StableDiffusionPAGPipeline:text_to_image
StableDiffusionPipeline:text_to_image
StableDiffusionXLPAGPipeline:text_to_image
StableDiffusionXLPipeline:text_to_image
StableDiffusionXLTurboPipeline:text_to_image
Wan22Pipeline:text_to_video
WanVACEPipeline:text_to_video
```

The exact 92 conditioned candidates and their required inputs are maintained in the candidate contract rather than duplicated here. Major cohorts include 17 built-in media operations; SD/SDXL edit, ControlNet, PAG and adapter variants; FLUX edit/control/inpaint/outpaint variants; Qwen editing/control/layered workflows; LTX/Wan video conditioning; audio continuation/repaint/variation; image depth; speech; and the dedicated video upscaler.

#### 4.2.1 Janus bounded qualification checkpoint (2026-08-22)

The cached `deepseek-community/Janus-Pro-1B` family remains generic through
`LoadAnyToAnyModel` and `GenerateAnyToAny`; no Janus-specific Studio node was
introduced. Official model-card and Transformers documentation established the
native 384-by-384, 576-token, BF16 sampled recipe. Text generation completed but
failed the reviewed output contract (repetitive, truncated headings instead of
five requested bullets), so it is quality-blocked without another prompt retry.

Image qualification exposed three distinct upstream development-pin boundaries:
the Janus static-cache call omitted the newly required `prefill_chunk_size`, the
pipeline postprocessor passed `PIL.Image.Image` as an unsupported tensor type,
and the NumPy postprocessor received a ROCm tensor without a host copy. Each
boundary received one source-driven finite compatibility fix with focused
contract tests. The third fix is implemented and tested, but another live run is
not authorized in this bounded sequence. The workflow is therefore
`technical_blocked_pending_later_canary_authorization`; it must not be retried
unchanged or counted as an asset. No media was emitted by any of the three
technical attempts.

The next cached Transformers-family canary, SmolVLM 256M image-to-text, used the
workspace-owner-approved AuraFlow rainy-shelter image and the official BF16,
chat-template, deterministic recipe. It completed in 14.6 seconds on ROCm and
proved the generic image-to-text graph executes, but its four-sentence response
violated the requested one-sentence format, contained “an red umbrella,” and did
not explicitly state that it was raining. The output was rejected and removed;
the exact task, input/output hashes, text, and runtime measurements are retained
in the rejection receipt. No unchanged retry is allowed.

### 4.3 Image and video upscaling

At least one canonical workflow already exists for each:

| Capability | Canonical workflow | Graph | Model/input state | Remaining publication work |
| --- | --- | --- | --- | --- |
| Image upscaling | `SpandrelImageUpscale:image_upscale` | `data/graphs/studio/spandrel-image-upscale/image-upscale.json` | Exact x2 model cached; live ROCm run and comparison candidate complete | Workspace-owner quality and rights review; broader platform qualification. |
| Video upscaling | `SpandrelVideoUpscale:video_upscale` | `data/graphs/studio/spandrel-video-upscale/video-upscale.json` | Exact x2 model cached; 81-frame motion-rich live ROCm run and comparison candidate complete | Workspace-owner quality and rights review; broader platform qualification. |

`BuiltinImageOperation:image_upscale` remains a deterministic Pillow interpolation utility and is not presented as model-backed super-resolution. Both model-backed upscaling templates use the same generic node family and exact app-managed Real-ESRGAN x2 artifact; no model-named execution node was added. Their remaining work is human approval, rights closure, public visibility, and broader hardware qualification—not another generation run with the same recipe.

## 5. Live cache and disk state

Snapshot taken 2026-08-21 from the app endpoints:

- 93 indexed Hugging Face cache entries;
- 87 complete and 6 incomplete;
- approximately 1.713 TB reported cache bytes;
- filesystem free: 106,983,280,640 bytes (about 99.6 GiB);
- reserve: 68,719,476,736 bytes (64 GiB);
- current headroom above reserve: about 38.3 GB;
- download queue: idle, zero queued.

Earlier audit data is not blindly carried forward. `google/diffusiongemma-26B-A4B-it`, previously a dependency-zero deletion candidate, is no longer present in the live cache. It must not be redownloaded until it has a bounded executable contract. The change happened outside this execution lane and is recorded as external state, not as a completed MoDiff eviction action.

Six inactive `.incomplete` blobs (mainly Marigold/LTX) total about 1.946 GB. They may be removed only through an app-supported stale-download cleanup, never manual unlinking.

## 6. Cached-first execution and turnover order

The objective is not “run every cached model once.” It is “close every in-scope workflow for a repository, preserve evidence, then reclaim the repository.”

### Cohort 0 — preserve approvals and close receipts without rerunning

First apply the user’s approvals to exact receipts, research/record model and source-media rights, fill duration/peak-memory/output-hash fields, and publish or archive immutable evidence. Do not spend GPU time replacing approved bytes.

Best high-value sole-workflow repositories currently include:

| Repository | Live cache size | Workflow | Current quality state | Public migration dependency | Turnover action |
| --- | ---: | --- | --- | --- | --- |
| `fal/AuraFlow-v0.3` | 65.96 GB | `AuraFlowPipeline:text_to_image` | user approved | none | Close rights/receipt, then eligible. |
| `baidu/ERNIE-Image-Turbo` | 31.63 GB | `ErnieImagePipeline:text_to_image` | user approved | none | Close rights/receipt, then eligible. |
| `meituan-longcat/LongCat-Image` | 29.32 GB | `LongCatImagePipeline:text_to_image` | user approved | none | Close rights/receipt, then eligible. |
| `Photoroom/prx-512-t2i-sft` | 15.51 GB | `PRXPipeline:text_to_image` | user approved | none | Close rights/receipt, then eligible. |
| `zai-org/CogView4-6B` | 31.13 GB | `CogView4Pipeline:text_to_image` | existing campaign proof needs final review/receipt closure | none | Present exact asset, close or rerun once if rejected, then eligible. |

These candidates can potentially reclaim about 142 GB from the four already user-approved repositories before any additional generation. Exact sizes and revisions must be re-read immediately before deletion because cache state can change.

### Cohort 1 — finish one remaining workflow, then reclaim

| Repository | Size | Completed side | Remaining side |
| --- | ---: | --- | --- |
| `Efficient-Large-Model/Sana_600M_1024px_diffusers` | 10.07 GB | Sana PAG approved | Quality-blocked after two researched text-to-image attempts; require a new official-recipe hypothesis before any new generation. |
| `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers` | 7.70 GB | Text-to-image approved | Edit quality-blocked after the two-attempt boundary; live deletion plan also found six saved-workflow dependencies, so retain. |
| `stabilityai/sdxl-turbo` | 6.94 GB | Technical execution only | Research and rerun once; current eyes/anatomy are rejected. |
| DreamLite Mobile repository | re-read live size | Text-to-image and edit approved | Close exact rights/metrics and confirm no other manifest dependency. |
| LCM DreamShaper repository | re-read live size | Text-to-image and edit approved | Close exact rights/metrics and confirm no public canary dependency. |

### Cohort 2 — cached new workflows with meaningful reclaim

After Cohorts 0–1, choose repositories where all remaining workflows can be bounded and where the resulting deletion creates material space. Examples include JoyAI Edit, OmniGen, HunyuanDiT variants, CogView3 Plus rework, Lumina2 rework, Kandinsky3 rework, and Z-Image’s four-workflow closure. Do not mix retry-prone GLM, AnimateDiff, ACE-Step, or static-video families into a clean turnover batch.

### Cohort 3 — public migration holds

Retain any repository used by the 44 stale or 7 first-example public templates until its exact canary/first example is accepted. Important holds include:

- `black-forest-labs/FLUX.1-schnell`: retain for the FLUX Schnell public canary and exact GGUF/base assembly evidence;
- FLUX Dev/Krea/Control/Fill/Kontext families: public stale/first-example dependencies remain;
- Qwen Image families: many public stale canaries remain;
- LTX 13B and Wan 1.3B/2.2 families: stale and first-example video templates remain;
- ACE-Step base and LoRAs: stale audio templates and two first examples remain.

The older `Lightricks/LTX-Video` cache (about 28.42 GB, revision `8984fa…97bb`) has no current 198-workflow or exact saved-workflow dependency in the previous audit, while current contracts use the 13B distilled repository. It remains a lower-confidence expert alternative. Re-audit source/catalog references before considering deletion.

## 7. Rejected-asset repair queue

Repairs are scheduled by root cause, not by filename order.

### Audio

- ACE continuation/repaint/variation/text-to-audio: use at least 30-second showcases (normally the reviewed 75-second source for edits), descriptive musical metadata, structured lyrics where appropriate, continuity/listening checks, and official ACE-Step duration/step guidance.
- AudioLDM2: name a recognizable sound scene, use the official 200-step/3.5-guidance baseline, generate bounded candidates, and select only if the intended scene is unambiguous.
- LongCat Audio: decide whether the example is music, speech, or singing; prompts and review labels must make that intent understandable.

### Video

- Automatically reject near-static outputs before human review using frame-difference/optical-flow thresholds plus contact-sheet inspection.
- AnimateDiff/PAG should use the official fine-tuned SD1.5 base and scheduler recipe. The current plain SD1.5 base is not the official quality recipe; do not rerun until the correct base is planned/downloaded.
- AnimateLCM likewise needs its official realism base; do not repeat the plain-base attempt.
- CogVideoX should use its official frame/step/guidance range and an explicit moving subject/camera prompt.
- LTX, Wan, and TI2V must visibly change in the card preview. Use longer, action-bearing prompts and preserve frame count/FPS; dedicated Wan video routes are preferred over VACE text-to-video when appropriate.
- ShapE needs a larger, inspectable orbit preview.

### Image and utility

- Preserve user-approved AuraFlow, Chroma, DreamLite Mobile, ERNIE, Flux Krea, FLUX GGUF, HunyuanDiT, LCM, LongCat Image, Marigold accepted variant, PRX, Qwen, Sana Sprint, SD PAG, and other explicitly approved bytes.
- Rework DreamLite base, Flux2 Klein, CogView3 Plus, Kandinsky3, Lumina2, SD/SDXL/SDXL Turbo, Sana theme duplication, and Z-Image with model-specific official defaults and new prompts.
- DDIM/DDPM/Consistency outputs remain technical capability proofs. Their native tiny resolutions must not be upscaled and mislabeled as quality showcases; use an explanatory comparison card or keep them out of the public Gallery.
- Built-in transforms must be before/after cards. Do not stage a bare output as if it demonstrated the operation.

## 8. Download sequence after safe reclaim

Every item below is conditional on two fresh app plans, available disk, exact license state, and the repository still being absent.

1. Correct official AnimateDiff/AnimateLCM base models needed to repair rejected videos.
2. ShapE image-to-3D dependency, previously planned at about 1.69 GB, to close the conditioned ShapE workflow.
3. Stable Video Diffusion (about 4.51 GB) and Stable Audio (about 5.35 GB), only after license/token gates are satisfied.
4. Medium new-video batch: Latte (about 23.62 GB), Allegro (about 25.29 GB), and FramePack (about 25.75 GB), one repository at a time.
5. Mochi (about 40.03 GB) and Nucleus (about 51.66 GB) after another turnover checkpoint.
6. Large Wan additions only with dedicated space: FLF about 90.10 GB and T2V A14B about 126.20 GB.

`city96/FLUX.1-schnell-gguf` Q4_0 is already cached and its visual output was user-approved. Do not plan another GGUF download for the same proof.

## 9. Engineering work remaining

### P0 — reliability and truthfulness

- [x] Isolate browser/page failures between campaign templates.
- [x] Fix graph-finalization handoff races.
- [x] Require exact optional-runtime profile ID plus spec digest; remove the wrong-runtime bypass.
- [x] Preserve output MIME when exporting extensionless WebP results.
- [x] Fix notification/run navigation local-storage overflow and contain restoration errors.
- [x] Add compact workflow-library summaries and lazy exact-document loading so “My workflows” never transfers tens of megabytes just to render names. On the current 1,017-document library, the live metadata response is 223,879 bytes versus 38,359,684 bytes for the legacy full list (171× smaller); no summary contains a snapshot, and opening a row is browser-tested to fetch its one exact document.
- [x] Harden `DELETE /hf_cache/{revisionHash}` with queue/download interlocks and dependency closure checks. The app now requires a fresh immutable deletion-plan hash, blocks active/queued graphs, downloads, Gallery installs, open canonical receipts, and saved-workflow references, and shows the exact repo/revision/size before confirmation.
- [x] Add app-supported incomplete-download cleanup. Model Manager now obtains a hash-bound plan that includes only regular `models--*/blobs/*.incomplete` files older than one hour, blocks active/queued graphs, downloads, and Gallery installation, and revalidates each exact path/size/mtime under the cache mutation lock before unlinking. The first live execution removed six 9–38-day-old partial files and reclaimed 1,945,039,075 bytes; a fresh plan reports zero eligible files, while Marigold and both affected LTX snapshots remain complete and repair-free. No model revision was deleted.

### P1 — migration and template contracts

- [x] Prove all 77 public definitions resolve and all 198 canonical graphs are structurally healthy.
- [ ] Run 44 exact stale canaries, cached families first.
- [ ] Create the seven first public examples.
- [ ] Restore/verify the pinned Gallery provenance dataset before release-contract generation.
- [x] Add a live-backend family canary proving exact cached model selections replace backend parameter contracts while node modules remain generic. Hunyuan→PixArt, CogVideoX→Sana Video, and AudioLDM2→LongCat Audio all produced different finalized node contracts through `modules.DiffusersImage`, `modules.DiffusersVideo`, and `modules.DiffusersAudio`; Shap-E, Real-ESRGAN, and SmolLM2 additionally finalized through generic `modules.DiffusersThreeD`, `modules.Spandrel`, and `modules.HuggingFaceTransformers` nodes. The expanded focused Playwright canary passed in 1.3 minutes without executing a model. The Transformers catalog's package-level `present_unqualified` observation is intentional; the execution proof used the qualified, cutover-ready exact active `huggingface-transformers-main-96fe6dce-peft-0.20.0` profile and matching spec digest.
- [ ] Confirm image and video upscaling appear in the intended browser surfaces and publish one accepted template/card for each.

### P2 — asset quality system

- [x] Separate technical success, quality approval, rights approval, and publication state.
- [x] Ingest user feedback into a machine-readable quality ledger.
- [x] Refuse synthetic two-second audio placeholders and classify static videos.
- [x] Remove the old rejected/withdrawn campaign bytes from the visible review queue while retaining recoverable Trash copies and index audit records.
- [x] Require the exact media-kind automated screen plus a hash-bound, all-pass Codex screen before an item can enter the visible human-review index. Raw campaign bytes remain explicitly unreviewed and cannot be staged without both gates.
- [ ] Reconcile generated media, frozen-checkout acceptance JSON, user decisions, and the main review index into one authoritative receipt; fail staging when they disagree. Exact Studio-output/user-decision reconciliation is implemented and applied to the first four turnover candidates. Frozen acceptance ingestion now requires an exact workflow + media SHA-256 + task ID match, binds the immutable receipt hash, and closes task/runtime metrics only when those fields are present; focused tests pass. Applying it to AuraFlow, ERNIE, LongCat Image, and PRX proved that their frozen campaign receipts describe different bytes/tasks, so no evidence was inherited. Exact task-metric recovery for the approved bytes and legacy MIME filename normalization remain.
- [ ] Add comparison-card authoring for built-in transforms and upscalers.
- [x] Require every newly staged generation receipt to bind an exact dossier under `review-pending/research`, at least one official-source finding, the dossier SHA-256, and the canonical recipe content hash; staging fails closed when the quality report does not bind that same dossier.
- [ ] Add duplicate-theme and duplicate-source checks across the Gallery.

### P3 — upstream and runtime compatibility

- [x] Current generation runtime reports Diffusers `0.40.0.dev0` and Transformers `5.16.0.dev0`; the app is not simply stuck on an older released Diffusers version.
- [ ] Pin the exact upstream commits, regenerate source/capability ledgers, and run the full generic action matrix against those commits.
- [ ] Repair GLM’s Float/BFloat16 component boundary with a component-level dtype canary before another full generation.
- [ ] Keep Stable Audio 3 classes contract-only until an official Diffusers-format artifact exists.
- [ ] Requalify Wan/LTX exact routes under the pinned upstream runtime.

### P4 — quantization and optional runtimes

- [x] One exact FLUX GGUF Q4_0 path works and has user-approved visual output.
- [x] Optional-runtime readiness requires the exact active profile/spec pair.
- [x] Prevent TorchAO or Quanto from appearing usable when their app-delivered runtime is unavailable. The live optimization catalog reports both unavailable on the current ROCm runtime, all 17 execution profiles that declare either mode publish neither in `available_expert_quantization_modes`, Studio renders and validates against that filtered list, and 28 focused tests plus 55 subtests pass.
- [ ] Ship platform-specific immutable TorchAO/Quanto runtime profiles, qualifying one backend/platform at a time.
- [ ] Add one bounded bitsandbytes path on NVIDIA before expanding to AWQ/GPTQ.
- [ ] Document supported quantization by exact family, platform, downloadable artifacts, installer path, and fallback—not merely by library name.

### P5 — generic client and documentation

- [ ] Keep model and mode definitions backend-discovered; remove remaining duplicated client catalogs where the backend can be authoritative.
- [x] Verify dynamic node parameters for at least one model from each image, video, audio, 3D, Transformers, and upscaler family. The live backend canary now covers all six broad families and asserts that every finalized execution graph retains its generic module boundary.
- [ ] For every exposed open-source model, document license, revision, approximate download size, app download steps, required auxiliary artifacts, supported actions, recommended defaults, hardware/runtime qualification, and known limitations.
- [ ] For every exposed Diffusers/Transformers action, link the official API/model documentation and describe MoDiff input/output mapping.
- [ ] When the app cannot download an artifact automatically, show exact guided steps in-product; do not leave a silent missing dependency.

## 10. Immediate execution sequence

1. **Completed:** fix and verify notification-to-workflow navigation against a real recent run.
2. **Partially completed:** AuraFlow, ERNIE, LongCat Image, and PRX now have exact byte-, task-, graph-, repository-, and revision-bound quality receipts. Close the explicitly recorded rights, task-metric/runtime-fingerprint, and legacy MIME filename gaps without rerunning approved media.
3. **Completed:** deletion planning/interlocks are implemented and tested. A live plan for Sana Sprint enumerated two open canonical dependencies and six saved workflows; the exact DELETE was refused with HTTP 409 and the cached revision remained complete. Only revisions with a fresh green plan may now be evicted.
4. In parallel with receipt work, run cached public migration canaries that do not belong to a retry-prone family.
5. Sana 600M and Sana Sprint were researched and bounded. Sana text-to-image failed two quality attempts. Sana Sprint edit attempt 1 preserved too much and failed the brief; the source-researched 0.6-strength correction executed both denoising steps but destroyed source composition. Both Sana Sprint edit outputs were rejected and removed to recoverable Trash. Do not retry either recipe without a new source/brief qualification study or a different exact model.
6. Create the image-upscale and video-upscale comparison assets using the already cached upscaler.
7. Work the seven first public examples, prioritizing cached ACE/FLUX/Wan dependencies while keeping video/audio quality gates strict.
8. Reclaim space at each fully closed repository checkpoint; run two fresh download plans and acquire only the next bounded batch.
9. Repair rejected families only after their research/engineering prerequisite is met.
10. Continue through the hidden candidates in cache-first, repository-complete cohorts.

## 11. Definition of done

A template is complete only when all applicable items exist:

- canonical generic graph and current graph hash;
- exact template/catalog lock;
- successful real-run receipt on a named runtime/hardware profile;
- model and auxiliary artifact revisions;
- prompt/settings/input provenance and recipe-research references;
- output hash, MIME/extension agreement, duration/dimensions/FPS, execution time, and peak memory;
- automated media-kind screen;
- explicit human quality decision;
- explicit rights/provenance decision;
- published Gallery asset or documented reason it remains hidden;
- no unresolved migration or dynamic-field mismatch.

A model revision is turnover-complete only when every dependent template/workflow is complete or explicitly blocked, durable evidence is outside the cache being removed, the app dependency check is green, and post-delete verification confirms both reclaimed space and retained Gallery assets.

## 12. Estimated effort

These are active engineering/GPU-time estimates, not promises about unattended wall time:

- notification fix and live gate: complete;
- receipt closure plus safe deletion interlock: 4–8 hours;
- first high-value turnover checkpoint: 0.5–1 day, mostly rights/receipt verification;
- cached public stale canaries: 1–3 days depending on video/audio runtime;
- seven first public examples: 2–5 days because ACE/Wan quality review is expensive;
- all 147 hidden candidates: multi-week qualification program, dominated by the 92 conditioned inputs, new downloads, video runtimes, and human review;
- upstream pin, GLM dtype, first app-delivered TorchAO/Quanto profile, and documentation baseline: 2–5 engineering days, with NVIDIA-only BnB qualification requiring the relevant host.

The queue should be re-estimated after every repository turnover, upstream pin change, or batch of human decisions. The plan must never hide a quality or dependency blocker by counting a technically generated file as a finished asset.
