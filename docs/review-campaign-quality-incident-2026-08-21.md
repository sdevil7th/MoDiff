# Review campaign quality incident — 2026-08-21

## Outcome

The 2026-08-21 `review-pending` batch must not be treated as an approved asset set. It mixed raw runtime
canaries, built-in operation proofs, native low-resolution samples, and intended Gallery showcases in one
surface. The workspace owner reasonably interpreted every visible file as pre-screened showcase work.

All generation is paused until the separation and quality gates below are enforced. Existing files remain
available as diagnostic evidence, but rejected or unreviewed files must not be published, registered, bound to
a public template, or used to represent model quality.

## What failed in the process

1. **Raw execution was confused with quality approval.** A completed task proved only that a graph ran and
   produced decodable bytes. It did not prove that the result satisfied the prompt, demonstrated the workflow,
   moved like a video, sounded intentional, or was suitable for a template card.
2. **Already rejected videos remained in the review surface.** The internal video screen classified 14 of 15
   clips as `rerun_required`, but the retry ledger did not consume that screen and the files remained beside
   review candidates.
3. **Audio integrity checks were too narrow.** They checked decoding, peaks, silence, and tails. They did not
   judge composition, source suitability, prompt recognizability, musical development, or splice continuity.
4. **Generic fixtures were used where the workflow required meaningful source media.** ACE-Step variation and
   repaint were driven by a two-second synthetic tone. A technically successful variation of that fixture could
   never demonstrate a useful musical transformation.
5. **Campaign “research” measured prompt length instead of recipe correctness.** It compared word counts with
   public prompts and catalog titles, but did not qualify the exact upstream base checkpoint, scheduler, frame
   count, inference steps, guidance, source fixture, duration, or official example for each family.
6. **Representative upstream settings were knowingly reduced.** CogVideoX ran a 25-frame/25-step review recipe
   even though its contract recorded the representative 49-frame/50-step recipe. That optimization was not
   suitable for quality qualification.
7. **Built-in transformations were staged as standalone outputs.** An adjustment, channel extraction, trim,
   loudness match, or join only makes sense with its input and an explicit before/after explanation. The audio
   join and loudness files also reuse the same ACE source waveform; they prove the operation, not unique content.
8. **Native-resolution unconditional checkpoints were treated as card assets.** The DDIM and DDPM checkpoints
   are natively 32×32 and the consistency checkpoint is 64×64. Those are legitimate technical canaries, but
   enlarging or displaying them alone does not create a useful Gallery showcase.
9. **Theme and semantic diversity were not gated.** Multiple reading rooms, botanical studies, umbrellas, and
   similar compositions reached the batch without a similarity or concept-diversity screen.
10. **The reviewer was asked to perform work that automation and the first-pass screen should have removed.**
    Static videos, two-second audio, tiny native samples, and obvious prompt failures should never have consumed
    human review time.

This was a campaign design and review-state failure, not evidence that generic nodes are the wrong architecture.
No model-specific frontend node fork is an acceptable fix. Model-specific loading and execution details remain in
backend adapters, runtime profiles, catalogs, and authored graph data; the Studio nodes remain generic.

## Upstream recipe evidence

The correction is based on primary upstream documentation, not prompt guesswork:

- ACE-Step 1.5 generates 10-second to 10-minute stereo 48 kHz audio. Its official guidance calls for a
  descriptive genre/instrument/mood/tempo prompt and structured lyric sections. The turbo checkpoint is an
  eight-step guidance-distilled recipe. Source-audio modes require real preprocessed audio, not a synthetic
  two-second tone: <https://huggingface.co/docs/diffusers/api/pipelines/ace_step>
- AudioLDM2 defaults to 200 inference steps and guidance 3.5, and supports multiple generated waveforms for
  selection. Its examples use concrete identifiable sound events rather than an abstract music brief:
  <https://huggingface.co/docs/diffusers/api/pipelines/audioldm2>
- AnimateDiff's official Diffusers recipe uses the v1.5 motion adapter with a finetuned Stable Diffusion 1.5
  base, DDIM with `clip_sample=False`, `timestep_spacing="linspace"`, `beta_schedule="linear"`, 25 steps,
  guidance 7.5, and 16 frames. The documentation explicitly warns that the scheduler beta schedule matters and
  shows reference outputs: <https://huggingface.co/docs/diffusers/main/api/pipelines/animatediff>
- AnimateLCM's published recipe uses six steps, guidance 2, and LoRA scale 0.8. A nearby step count is not enough
  if the base checkpoint, adapter configuration, or fixture is wrong:
  <https://huggingface.co/wangfuyun/AnimateLCM>
- CogVideoX's representative Diffusers call uses 49 frames, 50 inference steps, and guidance 6. The previous
  25-frame/25-step campaign shortcut is withdrawn from showcase qualification:
  <https://huggingface.co/docs/diffusers/main/api/pipelines/cogvideox>
- LTX 0.9.8 13B distilled examples and model card are the visual reference for the exact cached family; the
  input image or video must itself contain a composition that can yield readable motion:
  <https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled>
- Wan's documented representative temporal recipe uses 81 frames, and its own negative guidance explicitly
  rejects static/still-picture output. Motionless Wan results cannot be accepted as video:
  <https://huggingface.co/docs/diffusers/api/pipelines/wan>
- LongCat Audio's official proof is a short concrete sound scene. A short upstream capability proof may remain a
  technical canary, but it must not be marketed as a 30-second music showcase:
  <https://huggingface.co/docs/diffusers/main/api/pipelines/longcat_audio_dit>

## Workspace-owner review disposition

The comments supplied on 2026-08-21 are quality decisions only. They do **not** imply rights approval,
publication approval, Dataset upload, or Gallery registration.

### Quality accepted for now

- AuraFlow text-to-image
- Chroma text-to-image
- DreamLite Mobile edit and text-to-image
- ERNIE Image text-to-image
- FLUX Krea text-to-image
- FLUX Schnell GGUF Q4_0 text-to-image
- HunyuanDiT text-to-image
- Latent Consistency Model edit and text-to-image
- LongCat Image text-to-image
- Marigold depth using `campaign-marigold-dreamlite-depth-v1.png` only
- PRX text-to-image
- Qwen Image Modular text-to-image
- Sana Sprint text-to-image outputs
- Stable Diffusion PAG text-to-image outputs
- Built-in audio join, loudness match, and trim are accepted as **operation correctness proofs** only. Join and
  loudness match are transformations of the same ACE source and are not unique Gallery showcase content.

### Conditional: do not publish before an improved replacement or explicit keep decision

- CogView3 Plus: composition is useful; malformed umbrella and handle require a corrected output.
- Sana text-to-image: quality is acceptable, but the repeated theme must be replaced with a distinct concept.
- Sana Video: keep only if one researched improvement cannot materially improve it.
- Stable Diffusion XL PAG: improve the animated-looking face; otherwise return for an explicit keep decision.
- Stable Diffusion XL: improve overall quality.
- Wan VACE and Wan Video: visually promising, but require enough motion to read as video in a card preview.

### Rejected or withdrawn from showcase review

- All four reviewed ACE-Step campaign outputs: continuation, repaint, variation, and text-to-audio
- AnimateDiff PAG, AnimateDiff, AnimateDiff video-to-video, and AnimateLCM
- AudioLDM2 text-to-audio
- Built-in image adjustment and channels; all other built-in image/data/video canaries are withdrawn from the
  showcase queue and require labeled before/after presentation if ever shown
- CogVideoX text-to-video and video-to-video
- DreamLite text-to-image
- FLUX 2 Klein text-to-image
- Kandinsky 3 text-to-image
- LongCat Audio text-to-audio as a Gallery example; retain only as a short technical proof until a clear purpose
  and appropriately labeled presentation exist
- LTX long multi-prompt image-to-video and all reviewed LTX image/reference/text/video-to-video clips
- Lumina 2 text-to-image
- The second generic Marigold depth output
- Consistency, DDIM, and DDPM standalone native-resolution images as Gallery card assets
- Shap-E preview at the reviewed size
- Stable Diffusion 1.5 control/edit/text outputs
- Stable Diffusion XL Turbo outputs
- Wan TI2V
- Z-Image Modular

## Gates required before another human-review batch

### Common

- Raw execution outputs go only to `review-pending/raw-campaign/`.
- A successful backend task never changes review state by itself.
- Every shortlist entry must preserve workflow ID, exact model revision, prompt, negative prompt, seed, all relevant
  inference parameters, source-fixture hashes, runtime fingerprint, output hash, and the upstream recipe reference.
- A rejected receipt remains authoritative until a **new task ID and new output hash** pass all gates. Merely finding
  an existing file cannot clear it.
- A batch presented to the workspace owner contains at most five independently screened candidates.

### Images and depth

- A generative Gallery image must be at least 512 px on its shorter edge unless the workflow's explicit purpose is
  low-resolution generation.
- Native 32×32/64×64 unconditional outputs stay technical canaries. If documented in the UI, show them in a labeled
  model-explanation panel rather than pretending they are high-resolution examples.
- Run subject-count, blur, obvious anatomy/geometry, prompt-semantic, and perceptual-similarity checks before visual
  shortlisting. Similar concepts already represented in the Gallery require a genuinely different replacement.
- Built-in image operations require input/output comparison, operation label, and visible change; output alone is
  not reviewable evidence.

### Video

- Decode all frames and verify dimensions, duration, frame count, and frame rate.
- Reject static holds using adjacent-frame, end-to-end, active-window, and low-motion-frame measurements. The exact
  template threshold may be stronger, but no model video passes with negligible foreground/camera motion.
- Require a human-independent storyboard check: the prompted actor/object and motion must be visible and legible in
  the card crop, not merely inferred from noise or a blurred frame.
- Use the exact official or model-card recipe for the first canary. Any resource reduction must be qualified against
  that reference before it can replace it.
- Visually inspect the complete clip for flicker, morphing, identity drift, geometry collapse, and source timing.

### Audio

- Music and ACE-Step audio-to-audio showcases must be at least 30 seconds; continuation exports must include enough
  original context plus the generated tail to demonstrate continuity.
- Source and reference audio must be meaningful, rights-reviewed, workflow-specific assets. Synthetic tones are
  restricted to unit/runtime canaries.
- Check sample rate, channels, duration, clipping, silence, discontinuities, start and end transitions, and source
  boundary continuity before listening review.
- The prompt must identify what a listener should hear. Music needs structure, instrumentation, tempo/meter where
  useful, development, and a resolved ending. Sound-effect models need concrete events and spatial progression.
- When the pipeline supports multiple candidates, generate the documented candidate count and rank them before
  shortlisting one.

### 3D previews

- A turntable preview must be large enough for a card, show a complete object, use enough frames to reveal geometry,
  and be checked for watertightness/structural coherence separately from the rendered preview.

## Bounded repair order

1. Enforce raw-output separation, quality-only review state, and retry-ledger ingestion of video/audio screens.
2. Replace synthetic ACE source fixtures with the reviewed 75-second app-owned track; qualify each ACE mode with one
   focused canary before any full rerun.
3. Repair AnimateDiff's base-model and DDIM scheduler contract without adding model-specific frontend nodes.
4. Restore CogVideoX's representative 49-frame/50-step recipe and use a motion-readable prompt/fixture.
5. Run the existing LTX and Wan template motion gates against their current authored 81/121/161-frame recipes.
6. Regenerate only failed image families with distinct, model-appropriate concepts; reject malformed subjects before
   staging.
7. Re-present only 3–5 screened candidates with intent, settings, official reference, and any before/after input.

No family gets another full generation after a failed canary. A retry requires a documented change in checkpoint,
scheduler, fixture, prompt contract, or inference recipe; changing only the seed is not a researched fix.

## Compensation for the wasted review pass

- The rejected batch will be triaged and repaired without asking the workspace owner to rescreen technical canaries.
- Previously auto-rejected/static media will never be included in the next review batch.
- The next review surface will be a small shortlist with explicit purpose and provenance, not a filesystem dump.
- Every replacement will be paired with the exact prior failure and the concrete correction that allowed the rerun.
- Quality approval and rights/publication approval are now recorded independently, so the decisions above cannot be
  misrepresented as permission to publish.
