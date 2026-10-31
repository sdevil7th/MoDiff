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

Music controls such as lyrics, language, BPM and key appear only where the task
contract supports them. A nominal audio socket alone does not make those controls
valid for a sound-generation or speech model. AudioLDM2's candidate waveforms
remain available through Audio Variations. LongCat duration follows upstream
VAE-frame quantization; output is not padded to manufacture an exact duration.

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
