# Video prompts and motion examples

H3 and Pika can produce a visually detailed clip with little apparent motion
when the prompt asks for gentle drift and a steady viewpoint. Write a visible
action and a camera path explicitly. A large number of distinct decoded frames
does not establish that either is present.

For these models, use the documented `integrated_multimodal_description`,
`overall_soundscape` and `non_diegetic_music` sections. Describe events in
playback order, including their visible result. Give camera direction, travel
or angle, speed, and its ending view. Use a continuous take for camera motion;
use numbered shots and increasing cut times inside the duration for a genuine
change of scene. Sound should follow the action. See the [official Pika guide](https://github.com/Pika-Labs/Pika-H3-Flex/blob/64d81b2e98aa8aa1cf078b07a6ef3aaeae1033da/docs/prompt-guidance.md)
and the [pinned H3 guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).

For a first-frame-only example, put the documented opening-image instruction
before those sections and leave Last Frame unconnected. The existing first/last
stage accepts either or both frames. Two nearly identical endpoints constrain
the intended change; they should not be the only test of free camera movement.
Reference generation instead uses the documented six-section presentation,
including subject definitions and retention analysis. Declare the image's role
and which appearance details should remain. See the [reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).

Pika uses its official fixed six-evaluation sampler. Do not add Steps, CFG,
negative-prompt or invented motion-strength overrides. Its frame clock is
24 FPS; increasing export FPS only changes playback and sound alignment.
The revised examples use 124 frames (about 5.17 seconds) for short action and
192 frames (8 seconds) for two timed shots. Both match the upstream frame grid.
Keep the reviewed task adapter, precision and explicit offload recipe. The
longer example needs separate resource and output acceptance. [Diffusers API](https://huggingface.co/docs/diffusers/main/api/pipelines/pika_h3_flex)

[Motion recipe candidates](../plans/video-stages-20261101/motion-recipes.v2.json)
provide complete prompts and settings. Create them through ordinary operation
starters and save normal workflows. They are versioned creative descriptors,
not another execution system or a public-template qualification. Text examples
need no input media; conditioned examples take an operator-supplied matching
image. Existing saved values and original clips remain unchanged.

Review subject action, camera travel/parallax, requested cuts and scene identity,
temporal coherence and original audio separately. A motion or identity failure
stays a failed creative candidate even when encoding and playback succeed.
