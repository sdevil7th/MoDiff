# Diffusers execution qualification and model-rotation plan

Date: 2026-08-28

Status: active

## Objective

Qualify every reviewed Diffusers Cluster through the real frontend while using
local storage efficiently:

1. Use every exact, currently installed model before deleting it.
2. Produce reviewable, good-quality assets rather than low-step smoke outputs.
3. Prove the complete Cluster lifecycle for each route:
   `insert -> expand -> edit -> save -> refresh -> verify -> generate`.
4. Retain the workflow, immutable artifact revision, parameters, generated
   asset, hashes, browser evidence, and execution receipt.
5. Delete a model only after all of its routes pass and the retained outputs
   are approved or explicitly waived by the reviewer.
6. Install newly authorized families one at a time, qualify them, then rotate
   them out using the same process.
7. Keep legal, access, safety, hardware, and publication gates separate from
   structural Cluster support.

## Starting state

- Disk available at campaign start: approximately 156 GiB.
- Hugging Face cache at campaign start: approximately 1.56 TiB.
- Reviewed Diffusers Cluster definitions: 95 (94 official Modular Diffusers
  workflows plus one reviewed standard-Diffusers composite).
- Definitions with an exact pinned main artifact currently cached: 46.
- Unique cached pinned main artifacts used by those definitions: 17.
- Approximate storage occupied by those main artifacts: 736 GiB, excluding
  shared auxiliary ControlNet, IP-Adapter, LoRA, encoder, and upscaler assets.

The campaign must compute actual reclaimable bytes through the app before each
deletion. Displayed family totals are planning estimates, not deletion claims.

## Non-negotiable safety and evidence rules

- Do not use raw filesystem deletion for model rotation. Use Model Manager's
  exact, dependency-aware deletion flow and retain its receipt.
- Do not delete a model while any route that depends on it is unqualified or
  has an unresolved review decision.
- Do not treat a directory-name match as installation proof. Repository,
  40-character commit, reviewed file selection, and component identities must
  match the sealed admission.
- Do not accept a moving branch such as `main` in a promotion receipt.
- Do not silently accept a model license, privacy gate, AUP, commercial term,
  territory condition, or hosted-service condition for the user or company.
- A generic instruction to continue is not a revision-bound legal acceptance.
- Do not expose Prepare, Run, Model Manager download, `executable`,
  `liveProof`, or `autoEligible` claims until the exact route has passed its
  corresponding gate.
- Qualification assets remain private review evidence until manually approved.
- Rejected outputs are rerun using corrected prompts/parameters; technical
  qualification is not used to mislabel a poor output as showcase-ready.

## Review evidence layout

Campaign root:

`data/qualification/local-review/hugging-face-clusters/showcase-rotation-2026-08-28/`

Each definition receives a directory containing:

- generated image, video, or audio asset;
- source media, when applicable;
- saved workflow and post-refresh graph export;
- exact model and auxiliary artifact revisions;
- effective generation and execution parameters;
- frontend lifecycle receipt;
- generated-asset metadata and SHA-256;
- screenshots proving insertion, expanded controls, restored values, and run;
- contact sheet for video or waveform/spectrogram for audio;
- review decision with approve, reject, or waive status.

The campaign root also contains a machine-readable queue and a local review
page linking every pending asset.

## Phase 0 — freeze inventory and prepare the campaign

Target: 2–4 focused hours.

- [x] Export the exact installed repository/revision/file inventory.
- [x] Match every installed artifact to its sealed Cluster admission.
- [ ] Resolve shared auxiliary dependencies and actual reclaimability.
- [x] Create the dated campaign directory and review queue.
- [x] Define quality profiles from each pinned official model card and package
      contract. Profiles must include supported resolution/aspect ratio,
      sampling schedule, steps or sigmas, guidance, frame count/rate, duration,
      negative prompting, offload policy, and reproducible seed behavior.
- [ ] Create safe source fixtures for image, mask, control, driving-video,
      first/last-frame, reference, and audio-conditioned routes.
- [x] Confirm sufficient output storage without downloading another model.
- [x] Prove that a failed generation preserves the saved workflow and model.

Campaign implementation artifacts:

- `installed-artifact-inventory.json` freezes 17 exact installed artifacts and
  their 45 current definition consumers.
- `quality-profiles.json` records the official first-batch quality settings.
- `review-queue.json` is the private review queue. It keeps generated model
  outputs and optional delivery derivatives as distinct review items.
- `scripts/build_showcase_review.py` validates generated media and builds the
  campaign-local `index.html`, `review-index.json`, contact sheets, waveforms,
  media hashes, and technical checks without publishing anything.

## Phase 1 — first reviewable image/video/audio set

Target: first coherent manual-review batch within 6–12 execution hours.

### Image

- Qwen Image text-to-image.
- Use the official quality resolution and sampling profile rather than the
  prior minimum-cost qualification settings.
- Retain at least two seeded candidates and select without overwriting either.

### Video

- Wan Animate 2 or Wan 2.1, selected according to the strongest compatible
  prepared source fixture.
- Use a motion-bearing prompt/fixture, practical showcase resolution, enough
  frames to make temporal motion visible, and the official sampling schedule.
- Validate non-static motion numerically in addition to visual review.

### Audio

- MiniMax Music 3 at revision
  `fbdf52fbaaca799592917417eb05f1899f1255ec`.
- The user's exact-revision Community License/revenue acknowledgement is
  already recorded in the task history.
- Generate a meaningful-duration musical sample with lyrics/style controls,
  then validate duration, sample rate, channels, loudness, clipping, and
  non-silence.

Every route in this phase must pass the complete frontend save/refresh/run
lifecycle before its output is placed in the review queue.

### Phase 1 execution log

- Qwen Image text-to-image completed task `IcFmWpqEPzmE` at the pinned
  revision. The frontend restored the edited prompt, negative prompt,
  1328x1328 geometry, 50 steps, guidance 4, and seed 280828 after refresh; the
  collapsed and expanded exports matched. The 1328x1328 WEBP passed decoding,
  dimension, non-uniformity, and SHA-256 validation. Human review approved the
  image.
- Wan 2.1 text-to-video completed task `GEjlARR3FEej` after its full frontend
  lifecycle proof. Save and browser refresh restored the edited prompt,
  negative prompt, 832x480 geometry, 81 frames, 50 steps, guidance 5, 15 fps,
  and seed 280828; collapsed and expanded exports matched. The 5.4-second,
  81-frame H.264 MP4 passed decoding and non-static-motion validation and is
  pending human review.
- MiniMax Music 3 completed task `3DoJ-w8jP3cZ` at the acknowledged pinned
  revision. The visible frontend restored the edited style prompt, lyrics,
  30-second duration, 30 denoising steps, bfloat16/model-CPU offload, and seed 7
  after refresh; collapsed and expanded exports matched. The resulting
  30.02-second stereo 44.1 kHz WAV passed decoding, non-silence, clipping,
  duration, and SHA-256 checks. Human review approved the audio.
- Human review approved the Qwen image and MiniMax audio. The first Wan video
  was rejected because its distant subject made motion read as static and its
  quality-5 H.264 export was visibly pixelated. The rejected evidence is
  preserved under `wan-21-t2v-showcase/rejected-distant-dancer-seed-280828/`.
  The replacement keeps the model card's stable native 832x480 resolution,
  raises ordinary video export to the app's delivery-quality default of 8,
  uses guidance 6, and makes a large close subject traverse the frame and
  collide with multiple objects.
- The replacement Wan Cluster completed task `yXVYtz06P_nt` through the full
  frontend lifecycle. Its 81-frame, 5.4-second H.264 MP4 is 3,669,246 bytes,
  has a sampled motion mean of 58.0198, and preserves the new prompt, geometry,
  frame rate, guidance, steps, and seed across save and browser refresh. The
  native replacement is pending human review.
- An optional delivery derivative completed task `aRXn4cNS0Eql` through a
  separately saved and refreshed frontend `SpandrelVideoUpscale` workflow.
  Visible Expert Model Manager admitted
  `nateraw/real-esrgan@42efb9c3eeed1f5c0c8a626cf5f7f4481dfbb094`,
  the runtime released five resident Wan nodes before the model-family switch,
  and the app produced an 81-frame 1664x960 quality-8 MP4 in 91.58 seconds.
  The 15,739,015-byte derivative and native MP4 are separate review items.
- Human review rejected both the replacement 1.3B native clip and its 2x
  derivative because the apparent motion consisted of abrupt frame-to-frame
  jumps. Upscaling did not repair temporal coherence. Both review decisions
  are closed and preserved; neither output is eligible for promotion.
- The stronger cached Wan 2.1 I2V 14B Cluster completed task `Nop-smF5yYmu`
  through insert, edit, save, browser refresh, 15-block expansion parity,
  collapse parity, preparation, generation, decode, export, and frontend
  download. The official 832x480, guidance-5, 16-fps, 512-token controls and
  motion-focused prompt were retained. A measured 81-frame/50-step benchmark
  projected about 9.5 hours on this ROCm APU, so the review run used a valid
  49-frame (`4k+1`), 30-step bounded profile. The resulting 3.06-second H.264
  MP4 decoded to exactly 49 frames. All adjacent-frame continuity checks pass:
  no abrupt outliers, 0.999722 fifth-percentile adjacent structural
  correlation, and bounded transition dispersion. Automated screening left it
  pending because it cannot approve perceived smoothness or showcase motion.
  Human review then rejected it as effectively static: the camera,
  rain, and foliage motion requested by the prompt were not meaningfully
  visible.
- The review builder now reports motion presence separately from continuity.
  The rejected observatory clip remains free of abrupt transition outliers,
  but its 0.4393/255 median and 0.5495/255 p95 adjacent luminance change fail
  the meaningful-motion screen, so `technicalChecks.nonStatic` is false. This
  prevents accumulated distant-frame drift from masking a nearly frozen clip.
- Model Manager safely rotated out the closed Wan 2.1 I2V 14B and Wan 2.1 T2V
  1.3B snapshots after confirmation, preserving their workflows and evidence.
  The visible browser flow reclaimed 119,033,230,186 bytes and retained the
  runtime-release receipts under `storage-turnover/`.
- The replacement route is the exact Apache-2.0 Wan 2.2 I2V A14B artifact at
  `596658fd9ca6b7b71d5057529bbf319ecbc61d74`. Its source fixture is the guitar
  image used by the official Diffusers model card. The prompt follows the
  upstream I2V guidance: under 100 words, focused on visible hand/head motion,
  with a fixed camera so actual subject motion is auditable. The first
  17-frame canary was rejected before denoising by the production quality
  contract, which requires at least 81 frames for Wan I2V. The corrected
  qualification-only canary retains 81 frames while bounding sampling to 12
  steps; it runs before the full reviewed 832x480, 81-frame, 40-step,
  dual-guidance-3.5 recipe.
- The corrected canary passed Cluster insertion, parameter editing, Save and
  browser refresh, collapsed/expanded parity, and exact artifact preparation.
  During the first valid 81-frame execution, the pinned A14B pipeline reached
  about 112.8 GB anonymous RSS on this 121 GiB host. ROCm then requested more
  shared memory and the Linux OOM killer terminated the worker and headless
  browser before denoising produced an output. No canary asset or promotion
  receipt was created. The next run must use a proven bounded-memory execution
  recipe; repeating `model_cpu` offload unchanged is prohibited.
- Per-offload A14B resource admission is now enforced before model loading.
  CPU-resident BF16 routes require 160 GiB system RAM; the disk-group route
  requires 96 GiB system RAM, 24 GiB accelerator-accessible memory, and
  140 GiB free disk. Auto selects the first feasible reviewed mode instead of
  retrying an infeasible saved mode. Focused backend coverage (142 tests) and
  the complete client template suite (139 tests) pass. A new visible-frontend
  canary is currently running with `group_disk`: its 38 GiB offload store
  reduced worker RSS to about 5 GiB and reached the real 12-step denoising
  loop without repeating the OOM.
- The corrected Wan 2.2 A14B `group_disk` canary completed task
  `IZuRsOqU9AIJ` through visible insertion, edit, Save, refresh, expansion,
  collapse, preparation, generation, export, and frontend download. The app
  then released one resident model, six cached node objects, and all 34
  temporary offload files without an error. Its 832x480, 81-frame, 16-fps
  H.264 output is a valid non-static lifecycle/resource canary, but the
  12-step frames are overexposed and visually abstract; it is not eligible for
  showcase promotion. The exact A14B snapshot remains installed pending the
  review/turnover decision.
- The standard Diffusers Wan 2.2 TI2V 5B Cluster now reports its legacy cached
  snapshot against the exact catalog commit instead of briefly presenting an
  erroneous Install action. Its reviewed human label is retained consistently
  in the node library and inserted graph. The live browser proof has saved and
  refreshed `Wan 2.2 TI2V 5B Boxing Showcase`, preserved the expanded internal
  scheduler edit from flow shift 4.5 to the pinned value 5, explicitly proved
  expanded-state persistence, and submitted task `k6Q-Oe_fDM8n` with the full
  official 1280x704, 121-frame, 50-step, guidance-5, 24-fps recipe. The task
  completed through the visible frontend and exported a 9,673,535-byte MP4
  with media hash
  `sha256:bytes:069e3c204d8ffb6a38b50ad1dda587c076e245c783e078413fd6e048511a7255`.
  The receipt records collapsed/expanded graph equivalence and the pinned
  official Diffusers recipe. The asset is staged for human review;
  publication remains unchanged until approval.

Seven harness/runtime defects found by this real campaign were fixed:

1. Qwen's `text_encoder` is an upstream auto-selector; the editable official
   branch is `text_encoder/text_encoder`. The frontend E2E now targets the
   exact active branch instead of waiting on the sealed selector.
2. Model discovery treated the query string `refresh=false` as true and three
   Model Manager requests could rescan the large cache independently on the
   aiohttp event loop. Query parsing is now explicit and concurrent refreshes
   share one background actualization. The focused backend regression is
   green (24 tests and 29 subtests).
3. Custom-named Playwright output was not ignored and Vite watched long-run
   trace resources. That created tens of thousands of Git-visible temporary
  files and exhausted the host file-watcher limit. Custom result/report
  directories are now ignored, Vite excludes them from HMR watching, and
   long-running qualification can disable trace/video capture while retaining
   the compact review media and receipts.
4. Ordinary `modules.Video.Export` used a quality-5 default while the app's
   delivery exporters and video upscaler used quality 8. The ordinary exporter
   now inherits quality 8, with backend and live-browser regression coverage.
   The E2E assertion resolves immutable live defaults separately from explicit
   workflow overrides, matching the app's persistence contract.
5. Wan's active I2V Auto-block branches reuse `WanImageResizeStep` in both the
   image-encoder and VAE-encoder paths. Conditional expansion previously could
   not map those branch-container paths back to their two exact reviewed
   `get_workflow()` placements. The client now removes only immutable declared
   branch segments and requires a unique reviewed legacy-path match. The
   repeated-block regression and all 29 Cluster/node-library tests pass.
6. Long-running live-model tests had a one-hour active-execution budget even
   when their surrounding Playwright test explicitly allowed six hours. Normal
   live tests remain capped at one hour; only
   `MODIFF_LONG_RUNNING_QUALIFICATION=1` receives a ten-hour active-model
   budget within an eleven-hour model-I/O deadline. Ordinary live tests remain
   capped at one active hour.
7. Model Manager treated a missing Auto recipe as stronger than the exact
   artifact's installed status. That left reviewed Expert-only Cluster models
   labelled `Installing` after a successful download. Installation readiness
   now accepts either a runnable exact artifact or a ready Auto plan, with a
   regression proving the Expert-only case.
8. The Model Manager cache-turnover browser test still expected the retired
   confirmation copy. It now verifies the current safe-deletion dialog and
   exact-revision action. The run also fixed its visible pluralization from
   `dependencyies` to `dependencies`.

## Phase 2 — exhaust currently installed exact artifacts

Target: 3–5 focused working days, excluding reviewer response time.

### Current locally unblocked queue (2026-08-29)

The live library/cache join currently finds 46 graph-qualified Diffusers
admissions whose exact main artifact and every sealed model dependency are
already cached. Two Wan Animate 2 admissions remain hardware-blocked by their
mandatory compiled Flex Attention/remote-heavy qualification. The other 44 do
not need another model download or license acknowledgement before lifecycle
testing.

- The Wan 2.2 A14B I2V lifecycle/resource canary is complete and awaiting its
  review/turnover decision; its output is explicitly not a showcase candidate.
- The Wan 2.2 TI2V 5B standard-Diffusers route completed task
  `k6Q-Oe_fDM8n` after passing insert, edit, Save, refresh, expanded-state
  persistence, collapse, exact runtime preparation, generation, and export
  through the visible frontend. Its full-recipe MP4 is pending human review.
- 10 Qwen routes remain after the approved Qwen text-to-image route: five
  Qwen Image condition/edit routes, two Qwen Image Edit routes, two Edit Plus
  modes, and Layered decomposition.
  The next-wave preflight confirms that all four source/mask/control fixtures
  exist and that Qwen Image 2512, Image Edit, Edit Plus, Layered, and the exact
  InstantX ControlNet Union dependency are complete at their sealed commits.
  Every reviewed file selection matches its current app-owned cache plan; no
  Qwen download or legal acknowledgement is required for this batch.
- 8 FLUX routes are cached: FLUX.1 text/image, Kontext text/edit, and FLUX.2
  Klein/Base text/edit.
- 18 SDXL routes are cached with their sealed auxiliaries.
- 2 Z-Image routes, 1 ERNIE route, and 1 Wan FLF route remain in the local
  qualification/review queue.
- MiniMax Music 3 and Qwen text-to-image already passed the strengthened
  lifecycle and received asset approval. Their exact route-specific promotion
  receipts are now checked in and only those two admissions publish
  `liveProof: true`; executable and Auto eligibility remain dynamic
  runtime/resource decisions.
- The closed MiniMax family has a fresh dependency-protecting Model Manager
  deletion preview. Its approved WAV is retained independently, the exact
  28,517,608,999-byte revision has no open canonical or saved-workflow
  dependencies, and no deletion has been performed.
- Full regression, temporary offload cleanup, evidence reconciliation, and
  dependency-aware Model Manager turnover remain unblocked shared work.
- The installed Wan 2.2 TI2V 5B standard Diffusers route is represented by a
  first-party standard-pipeline Cluster without reusing the incompatible
  dual-expert Wan22 Modular block hierarchy. The current official
  `WanPipeline` call exposes text-to-video only, so the Cluster does not
  advertise native-code image conditioning.

### Cached-route campaign execution log (2026-08-29)

The current review bundle is
`data/qualification/local-review/hugging-face-clusters/cached-route-campaign-2026-08-29/`.
It contains only generated model media in its review index; browser screenshots
remain supporting evidence and cannot be selected as the route asset.

- All 10 selected Qwen routes passed the visible Cluster lifecycle. Layered
  decomposition retains each generated layer as a separate review asset.
- All 8 cached FLUX routes passed: FLUX.1 text/image, Kontext text/edit,
  FLUX.2 Klein text/edit, and FLUX.2 Klein Base text/edit.
- All 18 SDXL routes now have current
  insert/edit/save/refresh/expand/parity/generate receipts. The second batch
  added base text/image/inpaint, three ControlNet routes, ControlNet Union
  text-to-image, IP-Adapter text-to-image, and IP-Adapter + ControlNet Union
  inpainting instead of relying on loose historical outputs.
- Both Z-Image routes and the ERNIE Image text-to-image route passed through
  the visible frontend at their exact pinned revisions.
- Wan 2.1 FLF2V passed task `Y0rFaetIr2bk` through two endpoint-image edits,
  Save, refresh, 17-block expansion parity, and real MP4 generation. Its
  nine-frame/eight-fps output is qualification evidence, not a showcase claim.
- The completed private review page validates 42 of 42 generated assets across
  all 40 planned cached routes. These map to 39 definitions because Qwen Image
  Edit Plus has two admitted modes in one definition; the three Layered media
  items are retained separately. Every decision remains pending in this
  campaign; publication and model deletion remain unchanged.
- Repeated SDXL IP-Adapter workflow instances exposed duplicate
  `CLIPVisionModelWithProjection` loads with the same immutable Diffusers load
  ID. The loader now reuses one exact shared image encoder across Cluster
  instances and fails closed on ambiguous or dtype-incompatible resident
  state. Focused backend coverage proves reuse by separate node instances and
  the two failure paths.

The Wan 5B standard-pipeline Cluster implementation must therefore:

1. [x] add a reviewed `diffusers.composite` definition kind derived from the
   existing sealed `wan-22-ti2v-5b:text-to-video:v1` Studio execution spec;
2. [x] retain the exact loader, generate, export/preview roles and edges as the
   expandable internal graph, labelled as Diffusers components rather than
   pretending they are upstream Modular Diffusers sub-blocks;
3. [x] extend backend and frontend library validation with separate immutable
   composite block identifiers, hashes, and task membership;
4. [x] reuse the existing Cluster instance isolation, parameter override,
   save/refresh, fork-to-User-Node, materialization, and runtime-authority
   machinery;
5. [x] qualify the exact cached
   `Wan-AI/Wan2.2-TI2V-5B-Diffusers@b8fff7315c768468a5333511427288870b2e9635`
   artifact through the visible frontend; and
6. [x] keep native Wan-code image conditioning out of this Diffusers Cluster until
   the official Diffusers call itself exposes and MoDiff reviews that input.

The standard-composite implementation is covered by exact backend library,
runtime-authority, API, model-capability, and client parser contracts. Its live
frontend lifecycle and exact cached execution are complete. The remaining
route-specific work is human review, a definition-specific promotion receipt
if approved, and dependency-aware cache turnover after its resident family is
closed.

Run families in model-resident batches. Reviewer approval can overlap with the
next family, but deletion cannot.

| Order | Resident batch | Routes | Approximate main-model storage | Rotation rule |
| ---: | --- | ---: | ---: | --- |
| 1 | Qwen Image, Edit, Edit Plus, Layered | 10 | 215 GiB | Delete only after all ten route decisions are closed. |
| 2 | Wan 2.1 and Wan Animate 2 | 5 | 280 GiB | Retain all source fixtures and approved MP4s first. |
| 3 | FLUX.1, Kontext, FLUX.2 Klein/Base | 8 | 145 GiB | Preserve model-specific outputs and revisions separately. |
| 4 | Z-Image and ERNIE Image | 3 | 60 GiB | Do not merge their promotion evidence. |
| 5 | MiniMax Music 3 | 1 | 27 GiB | Preserve WAV/FLAC master plus preview. |
| 6 | SDXL and admitted auxiliaries | 18 | 10 GiB plus auxiliaries | Delete only auxiliaries with zero remaining consumers. |

For every definition:

- [ ] Insert the exact first-party Cluster.
- [ ] Expand the official block hierarchy.
- [ ] Modify at least one meaningful internal parameter.
- [ ] Save and refresh the browser.
- [ ] Verify all model, prompt, input, and execution values exactly.
- [ ] Generate through the visible frontend.
- [ ] Verify collapsed/expanded execution parity.
- [ ] Validate the media contract and output fingerprint.
- [ ] Add the result to the manual-review queue.
- [ ] Record approve, reject, or waive.
- [ ] Generate a definition-specific promotion receipt after approval.
- [ ] Run the exact Model Manager deletion preview after the family closes.
- [ ] Obtain deletion confirmation and retain the turnover receipt.

## Phase 3 — rotate through locally feasible new families

Target: 5–8 focused working days after required acknowledgements and access.

Install and qualify one family at a time after approved cached families free a
safe storage margin.

### Stable Diffusion 3

- Routes: text-to-image and image-to-image.
- Exact repository:
  `stabilityai/stable-diffusion-3-medium-diffusers@ea42f8cef0f178587cf766dc8129abd379c90671`.
- Repository size: about 31 GB; reviewed selected weights: about 15.5 GB.
- Required user/legal gate: affirmative Stability AI license acceptance and a
  statement that this qualification is non-commercial research and not a
  hosted/API/commercial use, or evidence of a separate applicable license.
- Engineering gates: authenticated component review, backend-owned bounds,
  optional-runtime admission, real AMD/ROCm execution, output review.
- Estimate after authorization: 1 focused working day.

### Krea 2 Raw and Turbo

- Exact repositories:
  - `krea/Krea-2-Raw@6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603`
  - `krea/Krea-2-Turbo@98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7`
- Each planned full app snapshot: about 62 GB; selective qualification target:
  about 48 GB.
- Required user/legal gates: Community License and incorporated AUP acceptance,
  confirmation of eligibility under the revenue term or separate enterprise
  authorization, and authorization to implement the required notices and
  content filtering.
- Engineering gates: immutable AUP receipt, downstream terms/notice UX,
  content-filter integration, backend bounds, optional runtime, AMD run.
- Estimate after authorization: 1–2 focused working days.

### HunyuanVideo 1.5

- Routes: text-to-video and image-to-video.
- Governing repository/license revision:
  `tencent/HunyuanVideo-1.5@9b49404b3f5df2a8f0b31df27a0c7ab872e7b038`.
- Diffusers artifacts:
  - T2V `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v@286be7ce72277246578a3e3cc2487e95ddae5bcf`
  - I2V `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled@854c04a4c8a53d990b418c7478f0802c0fc8c726`
- Selective disk estimates: about 68 GB T2V and 52 GB I2V.
- Required user/legal gates: license/AUP acceptance, confirmation that all
  execution and output use remains outside the excluded territories, MAU
  threshold eligibility or separate license, distribution notice, and
  generated-content disclosure.
- Engineering gates: exact territory disposition, backend resource bounds,
  AMD/ROCm execution, output review.
- Estimate after authorization: 1–2 focused working days.

### LTX-2.5

- Routes: text-to-video, image-to-video, condition, and in-context.
- Exact repository:
  `Lightricks/LTX-2.5-Diffusers@a6de4b5354f078db24d9cf4778c14846788aea3d`.
- The existing local entry is a zero-byte gated placeholder.
- Target distilled snapshot: about 72 GB excluding `transformer_full`; the full
  repository is about 110 GB.
- Earlier acceptance for LTX-2 revision
  `47da56e2ad66ce4125a9922b4a8826bf407f9d0a` does not automatically accept the
  separate LTX-2.5 Hugging Face access/privacy gate.
- Required user/legal gates: exact LTX-2.5 gate/privacy consent, LTX-2.x license
  acknowledgement for the pinned revision, revenue eligibility or separate
  authorization, and authenticated Hugging Face access.
- Engineering gates: authenticated index/partition review, exact distilled
  sigma schedule, AMD compatibility of the diffusion decoder and fetched
  NATTEN/kernel implementation, resource bounds, four live outputs.
- A convolutional decoder substitution is a modified User Node unless it is
  separately reviewed and admitted as a first-party Cluster workflow.
- Estimate after authorization: 2–3 focused working days; CUDA-only decoder
  behavior may move exact qualification into Phase 4.

## Phase 4 — remote-hardware or infrastructure-bound families

Target: 4–8 focused working days after suitable compute, access, region, and
spend authorization exist. Calendar completion is blocked until they exist.

### Cosmos 3

- Routes: 4 Distilled and 10 Omni workflows.
- No current Hugging Face click-through acceptance identified; model cards
  declare OpenMDW 1.1, global deployment, and commercial/non-commercial use.
- Engineering gates: immutable component descriptors, standard-index class
  resolution, qualified mandatory guardrail, AMD/runtime validation.
- Cosmos Nano planning envelope: about 64 GiB disk, 96 GiB RAM, and 24 GiB
  accelerator memory with offload. Attempt local qualification only after the
  component and guardrail gates close.
- Cosmos Super/Distilled 64B routes require remote multi-GPU hardware; the
  reviewed estimate is about 192 GiB system RAM and 128 GiB aggregate
  accelerator memory.
- Required external authorization: provider, allowed region, credentials,
  spending ceiling, retention policy, and permission to upload safe fixtures.

### MiniMax H3

- Routes: T2VA, FL2VA, and Ref2VA.
- Exact repository:
  `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`.
- Selective estimate: about 160 GiB; full repository is substantially larger.
- Planning envelope: at least 256 GiB RAM and about 192 GiB aggregate
  accelerator memory across four accelerators.
- Required user/legal gates: Community License/AUP acceptance, confirmation
  that execution, hosted access, and output use remain outside the United
  States, EU, UK, and Republic of Korea, revenue eligibility below the stated
  threshold or separate authorization, and distribution/AI-output notices.
- Required external authorization: permitted-region four-GPU environment,
  credentials, budget, storage, and retention policy.

### Ideogram 4

- Route: text-to-image.
- Preferred package-owned candidate:
  `ideogram-ai/ideogram-4-nf4-diffusers@1874bc70267ba2c823a7239e1d70dd308c8d64dc`.
- Visible repository size: about 16 GB.
- Required user/legal gates: exact non-commercial model agreement acceptance,
  confirmation that this use is non-commercial or separately licensed,
  incorporated AUP, downstream notices, human oversight, disclosures, and
  output safety filtering.
- Hardware gate: the official Diffusers route uses CUDA/NF4. The current
  AMD/ROCm machine cannot qualify that exact route. The publisher's FP8 variant
  is not supported by the reviewed Diffusers recipe.
- Required external authorization: CUDA environment, credentials, permitted
  use, budget, and retention policy.

## User and external blockers

At campaign start:

- The app reports no configured Hugging Face token. Gated downloads require the
  user to accept the repository gate on their Hugging Face account and configure
  that account's token through the frontend; credentials must not be sent in
  chat or written into receipts.
- Revision-bound acknowledgements remain required for LTX-2.5, SD3, both Krea
  variants, HunyuanVideo 1.5, MiniMax H3, and Ideogram 4.
- Remote-only work requires explicit provider, region, spend, fixture-upload,
  and retention authorization. Local full access does not grant cloud spend or
  third-party data-transfer authority.
- Public promotion requires the reviewer's definition-specific output approval.
- Physical macOS qualification remains separate from the AMD/ROCm Linux gate.

## Timeline

Assuming acknowledgements and remote compute arrive without delay:

- First image/video/audio review batch: within one working day.
- All 45 currently cached routes: 3–5 focused working days.
- New locally feasible families: an additional 5–8 focused working days.
- Remote/infrastructure families: an additional 4–8 focused working days.
- Full remaining Diffusers execution qualification: 12–21 focused working
  days, realistically 3–4 calendar weeks including downloads and review cycles.

A rejected image/audio asset typically adds 1–3 execution hours. A rejected
quality video may add 4–12 hours. Upstream AMD incompatibilities or unavailable
remote multi-GPU capacity can extend the corresponding family independently;
other admitted families should continue rather than waiting idle.

## Completion definition

The campaign is complete only when:

- every reviewed Diffusers definition has either an approved exact execution
  receipt or a clearly documented unresolved external blocker;
- all executable routes preserve edited values across save and browser refresh;
- expanded and collapsed executions are equivalent;
- approved assets and revisions are retained independently of model cache
  turnover;
- deletion receipts exist for rotated models;
- public execution flags match the approved evidence exactly;
- no policy-gated definition is advertised as executable by inference.

## Receipt-backed promotion and cached-route campaign — 2026-08-29

- [x] Added the checked-in, data-only
      `data/huggingface-cluster-promotion-receipts.v1.json` ledger and a strict
      backend validator. A receipt is accepted only when its definition,
      workflow, admission, immutable model revision, generated-media hash and
      size, runtime fingerprint, task identity, lifecycle proof, review
      decision, and publication claims all match exactly.
- [x] Bound the approved Qwen Image 2512 text-to-image and MiniMax Music 3
      receipts to their exact admissions. Only those routes now report
      `liveProof: true`; `executable`, `autoEligible`, and `galleryEligible`
      remain false until the separate runtime/resource and publication gates
      close. Other routes sharing the same model profile do not inherit the
      proof.
- [x] Preserved the complete Wan 2.2 TI2V 5B terminal execution receipt at
      `data/qualification/local-review/hugging-face-clusters/showcase-rotation-2026-08-28/wan-22-ti2v-5b-cluster/backend-execution-receipt.json`.
      The receipt records the exact task, artifact revision, runtime/resource
      fingerprints, phase timings, peak measurements, generated MP4 hash, and
      dedicated-worker exit without claiming an unavailable in-process cleanup
      receipt. Promotion remains pending reviewer approval of that output.
- [x] Re-ran the complete backend gate after the receipt binding, retained
      Modular cleanup correction, and shared IP-Adapter encoder fix: `2,265`
      passed, `54` skipped, and `6,730` subtests passed in 127.14 seconds.
      Focused IP-Adapter coverage passes `10` tests and `4` subtests.
- [x] Re-ran the complete client release gate. Formatting, linting, TypeScript,
      unit tests, production build, and bundle budgets pass; the complete
      mocked Studio browser suite passes `114/114` scenarios in 6.1 minutes.
- [x] Generated a dependency-aware MiniMax Music 3 deletion preview without
      deleting the model. The preview found no canonical or saved-workflow
      dependents, retains the approved WAV/hash, and remains subject to explicit
      turnover confirmation.
- [x] Removed eleven unpinned app-temporary media assets through the app's media
      cleanup path after verifying that none belonged to an active or queued
      task. Review assets and model snapshots were not touched.
- [x] Completed the visible-frontend Qwen batch covering ten exact cached
      routes: image edit, image-to-image, inpaint, edit inpaint, Edit Plus
      single and multi-reference modes, layered decomposition, and ControlNet
      text-to-image, image-to-image, and inpainting. Every route performed
      insert, edit, Save, browser refresh, expanded/collapsed graph parity, real
      generation, and media capture. The strengthened run completed five test
      cases (eight routes, with the first two already proven immediately before
      the rerun) in 11.8 minutes with five passing and two intentionally skipped.
- [x] Corrected conditional-workflow parameter placement for reviewed Modular
      Diffusers branches whose exact `get_workflow()` block has a distinct
      upstream id but the same class, kind, input/output, component, and config
      shape as its Auto-pipeline branch. The Qwen inpaint lifecycle now expands
      with its required `image_latents` parameter and executes successfully;
      unsafe class or shape mismatches still fail closed.
- [x] Corrected cross-model retained-component cleanup ordering. The app now
      releases the Modular `ComponentsManager` collection once before clearing
      loader nodes, avoiding repeated group-offload reconfiguration and garbage
      collection for every destructor. The exact Qwen Image to Qwen Image Edit,
      Edit Plus, Layered, and ControlNet transitions completed without the prior
      RAM/swap thrash.
- [x] Built the private cached-route review index with twelve actual backend
      image assets for the ten routes (including three layered outputs), each
      pinned to its task, definition, model revision, and exact media filename.
      Companion browser screenshots are excluded. Review decisions remain
      pending and no new public promotion is inferred.
- [x] Completed the post-change mocked Studio browser gate, reconciled the
      cached-route receipts into a 42/42 generated-asset review index, audited
      the two source repositories, removed disposable Playwright output, and
      created the dependency-aware Qwen deletion preview.
- [ ] Record reviewer decisions for the 40 cached routes, create promotion
      receipts only for approved definitions, then run fresh per-family Model
      Manager deletion previews. Do not delete any snapshot until that family
      is closed and turnover is explicitly confirmed.
