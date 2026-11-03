# Advanced video workflow examples

These ordinary MoDiff workflow packages combine the shared video stages with
existing image, video and audio operations. They remain editable graphs. The
stage count follows the job; it is not limited to four.

Import a package through **Gallery → Import workflow**, select your media in the
Load nodes, then save a named copy in **My workflows**. The source packages
contain no media or machine paths. Run must remain blocked until required inputs
are supplied. Select media available to the backend, and edit the subject and
reference descriptions in the prompt to match it.

| Package                                    | Inputs to select                                                | What the graph does                                                                                                                                                                                               |
| ------------------------------------------ | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pika-matched-keyframes.workflow.json`     | Opening and ending images                                       | Encodes both keyframes, generates a native video with audio, and exports it. Useful for comparing models with the same prompt and image bytes.                                                                    |
| `pika-mixed-references.workflow.json`      | Identity image, motion-reference video and audio-reference file | Trims the video/audio references to two seconds, assembles the three ordered reference kinds, generates joint video/audio and exports it. The selected audio may be an AV file.                                   |
| `pika-continuation-timeline.workflow.json` | A source AV file in both Load Video and Load Audio              | Extracts the actual last frame, generates an opening-image continuation, exports that native segment, then composes the source and continuation and joins/fits their soundtracks into a separate timeline export. |

The continuation's Frame Extract output is a collection. The existing Collection
Item node selects its first item before it reaches the image socket. No final
keyframe is forced. Video Compose uses no visual crossfade; Audio Join applies a
10 ms boundary fade and Fit Duration matches the composed video's duration.
The combined timeline is an edited sequence, not one native generation or a
model-produced scene cut. A matching opening frame does not guarantee matching
velocity or camera motion at the join.

The supplied continuation defaults were exercised with a 124-frame, 24 FPS source
and produce a second 124-frame native segment plus a 248-frame composed timeline.
For a different source duration, inspect the composition and audio timing rather
than assuming the resulting duration remains the same.

## Model and resource settings

These examples retain the immutable Pika H3 Flex model revision and its reviewed
official Diffusers provider. They require that exact optional runtime to be
explicitly installed and active, the corresponding model files to be available,
and the operator to accept the model's terms. Importing a package authorizes
neither installation nor downloads.

The examples use Custom memory, BF16, Group Disk, 832×480, 124 generated frames,
24 FPS and the official fixed six model evaluations. Do not add Steps, CFG or
negative-prompt overrides to Pika's fixed sampler. Required task adapters are
model-owned; they are not an optional user LoRA. See
[Video pipeline stages](../../docs/video-pipeline-stages.md) for the reviewed
offload, reference-preprocessing and runtime boundaries.

Large-model execution was checked on the Linux/ROCm instance with substantial
shared memory and SSD staging space. These packages do not establish Windows
fit, Auto support, other-hardware qualification or public Gallery readiness.
Use a resource plan appropriate to the actual machine, leaving headroom for
other applications.

## Quality and reuse

The keyframe example preserves a recognizable paper boat with gentle drift.
The mixed-reference example exercises all three input kinds and preserves
identity and coherent water detail, but its requested camera arc is weak.
It is a multimodal reference example, not a guarantee of strong camera motion.
The continuation retains the cyclist and pedalling with foreground parallax;
the transition is coherent, but the ending remains tighter than the requested
camera pullback. The native segment and composed timeline are separate exports.
Changing a prompt or source media creates a new creative result that needs its
own review.

Compare models using identical raw prompt strings and actual source-image
bytes. Keep each model's native geometry and sampler settings visible; these
examples are not equal-compute benchmarks. Retained MiniMax comparisons do not
require another MiniMax render.

Play the complete original output, inspect motion and continuity, and listen
to audio separately. Frame counts and muted playback do not establish audio
quality. Local technical/creative review does not grant publication permission.

Release the owning Model Setup and independent Load-node caches through their
ordinary scoped controls after use. Saved workflows, source media and durable
exports should remain available. Use **Arrange graph** after adding stages if
their controls overlap.
