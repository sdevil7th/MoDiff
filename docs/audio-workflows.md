# Audio workflow stages

Audio workflows use the same graph, ordinary nodes, Block interfaces, execution
and resource management as image workflows. Choose a task, select its exact model
and edit the controls declared by the backend. A model change must refresh the
task contract as well as the repository and revision.

## Keep the stages meaningful

| Route                                     | Real computation                                                        | Presentation                                                                                                                                                                                   |
| ----------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ACE-Step                                  | Load pipeline -> Generate -> Export                                     | Model Setup can combine the loader, its exclusive recipe/configuration and an authorized adapter. Generate retains prompt, lyrics and task controls. Export retains the existing audio player. |
| Stable Audio, LongCat AudioDiT, AudioLDM2 | Load pipeline -> Generate -> Export                                     | The same ordinary operations, with controls and output formats supplied by the selected backend adapter.                                                                                       |
| MiniMax Music3                            | Load models -> Semantic Generation -> Denoise -> Decode Audio -> Export | Decode Audio and its connected exporter can form Audio Output, keeping four visible stages and all five executable leaves.                                                                     |
| Speech recognition                        | Load model + Load Audio -> Transcribe -> Transcript                     | Audio stays a separate previewable input. The output is text/structured transcription, rather than an audio output group.                                                                      |
| Trim, join, loudness match                | Load Audio source(s) -> Process Audio -> Export                         | Ordinary deterministic operations; no model loader or adapter is required.                                                                                                                     |

The direct audio pipelines expose an integrated upstream Generate operation.
Splitting it into decorative Encode, Denoise and Decode nodes would not make its
internal computation editable. MiniMax's semantic generation is an actual stage
and must remain distinct from prompt encoding.

When the built-in execution recipe supplies audio placement, Model Setup presents
its Runtime Offload and Execution Device under **Device and offload**. The loader
retains its saved fallbacks for use when the recipe is disconnected. This display
order does not change the inputs passed to the audio pipeline.

Stable Audio uses its pinned Diffusers cosine scheduler. MoDiff supplies zero
for the final noise term when the schedule ends at zero. The upstream update
multiplies that term by zero, and a Brownian noise request outside the positive
sigma interval can exhaust recursion in `torchsde`. Earlier sampler steps and
the scheduler settings are preserved.

Grouping is an explicit, undoable presentation change using the existing Block
contract. Inspect implementation and Separate stages expose the original leaves.
Flattened execution, connected values and model ownership stay authoritative.
Existing saved workflows retain their presentation until deliberately changed.
Shared, ambiguous or custom branches require explicit structural editing.

## Sources and task controls

ACE variation, continuation and repaint require a connected source audio node.
Keep upload, preview, playback and seek available on that source. Continuation
can retain separate loudness-match and join operations; repaint keeps its bounded
time interval. These operations do not disappear when Model Setup is grouped.

ACE source-audio variation uses native `cover` conditioning through `src_audio`,
not the separate timbre-only `reference_audio` input. It requires both registered
`audio_tokenizer` and `audio_token_detokenizer` components in the selected
checkpoint. Their absence is an explicit compatibility error before source
tensor allocation or inference. The curated Turbo snapshot lacks those optional
components; its text generation, continuation and repaint routes are separate.
An alternate compatible checkpoint needs its own immutable revision, complete
artifact/resource admission and real source-conditioned output proof. Historical
reference-only receipts remain truthful records of the earlier behavior and
must not be relabeled as source-melody conditioning.

The curated source-cover template selects
`ACE-Step/acestep-v15-xl-base-diffusers` at
`8f40447d21ee59ad4dcee6c99ce248027410bfec`, with its native audio tokenizer and
detokenizer components. It retains the same generic Model Setup, Audio Input,
Generate and Audio Output stages. Its accepted recipe uses 50 steps, guidance 7
and Expert memory policy. This additional compatible artifact does not extend
the Turbo/Runware adapter-authoring grant or claim Base Auto/hardware
qualification. Older Turbo source-cover records stay historical.

An audio input can upload an MP4 containing a real soundtrack, as advertised by
its file picker. The upload preserves the original bytes and reports the actual
video-container kind; its audio preview extracts the soundtrack into a separate
managed cache. Silent video, corrupt media and images renamed as audio remain
rejected. Ordinary probes and original-file audio conversions keep their strict
kind checks; selecting audio for an upload does not relabel a video as audio.

Music controls such as lyrics, language, BPM and key appear only where the task
contract supports them. A nominal audio socket alone does not make those controls
valid for a sound-generation or speech model. AudioLDM2's candidate waveforms
remain available through Audio Variations. LongCat duration follows upstream
VAE-frame quantization; output is not padded to manufacture an exact duration.

For spoken LongCat showcases, enter the sentence to be spoken in Generate.
The reviewed checkpoint reads text aloud. An ambience description can
therefore produce narration of the description. Check the exact words and
listen for repeated or clipped endings before publishing a sample. The
[publisher's LongCat AudioDiT documentation](https://github.com/meituan-longcat/LongCat-AudioDiT)
describes the upstream speech model.

The selected audio contract must agree with the owner's literal repository,
immutable revision, pipeline and task. The revision remains on the model owner;
the audio signal itself carries the repository. Selecting an alternate compatible base must not
retain the previous model's hidden contract or source requirements.

## LoRA ownership

An adapter authoring declaration is permission to edit a particular model branch.
It is separate from broad model capabilities, installed artifacts, runtime
qualification and Automatic memory eligibility. The backend declares exact
tasks, repository/revision pairs, adapter node and supported row count; the
client validates that declaration against the current graph and node registry.

The selected ACE authoring route initially supports one owned adapter row. Edit,
replace and remove remain available at that capacity. Loading another adapter
with the ordinary Load Adapter action does not establish a blended chain: that
action activates its selected adapter. Multi-adapter blending needs an explicit
Set Adapters operation and its actual runtime/resource contracts. Other audio
families receive no adapter permission merely because they share a loader shape.

Legacy manual graphs remain editable through their ordinary nodes. Structural
removal can safely disconnect an owned row even after authoring permission is
withdrawn. Missing, stale, connected or ambiguous model identity must not grant
new automatic adapter edits. Direct audio adapter branches remain ineligible for
Automatic memory planning until their complete ownership and resource accounting
is supported; use the declared Custom/Expert path deliberately.

## Delivery and validation

AudioLDM2 needs a small compatibility bridge because the reviewed GPT-2 model
does not inherit the generation helper called by Diffusers. The bridge binds
the official Transformers helper, accepts only the reviewed releases
`5.16.0.dev0` and `5.18.0`, and verifies its source hash before use. It does not
install or activate a runtime. An unknown version or changed helper produces a
compatibility error instead of silently applying an unreviewed patch.

Audio Output uses the existing audio field/player and original file export.
Verify playback, seek, download and restoration against the same completed task.
Input previews and transcoded playback are not proof of the original output's
identity. Record measured duration, sample rate, channels, finite samples and
nonempty output separately from listening quality, lyrics, timing or edit seams.

Generation history records bounded lyrics, language, shift, BPM, key and time
signature only when the active audio adapter delivers them to the pipeline.
Inactive ACE controls on another audio family are not execution evidence. The
client preserves these scoped fields on history restore; older receipts remain
unchanged and missing controls are not reconstructed from saved form values.

ACE dispatch records also retain the actual task type and repaint interval, and
identify the consumed `src_audio` or `reference_audio` tensor after resampling,
stereo normalization and continuation padding. The node-local fields include
SHA-256 of contiguous raw bytes, shape, dtype, original device, byte count and
the pipeline-native sample rate. They do not change the legacy history summary.
Only plain dense CPU/CUDA tensors up to 32 MiB are copied for this evidence;
larger or unsupported inputs have an explicit capture status and no hash. Tensor
contents never enter history. A tensor identity establishes dispatch, not audio
quality or a claim that the generated edit preserved material outside its bounds.

Offload can reduce accelerator residency while increasing host-memory demand.
Supported options do not establish that a particular model, duration and recipe
fits a Windows machine. Quantized image support does not automatically authorize
quantized audio. Preserve creator settings and record any explicit test-instance
changes; a short smoke does not qualify the original full-length template.

Contract tests, browser interaction, real generation, independent repeat
inference, listening, memory qualification and public media approval are separate
proof levels. Follow [workbench acceptance](workbench-acceptance.md) and the
[engineering procedure](cluster-engineering-lessons.md). Video and mixed
audio/video workflows have their own contracts and are outside this audio work.
