# MiniMax, Pika and Wan image testing checklist

Latest user visual verdict, 3 November: seven displayed clips accepted, including
both cycling clips. Pika's referenced boat is rejected for unrealistic
paper-speedboat motion. Preserve that failed creative version; do not rerun the
other approved outputs solely for earlier framing/camera concerns. Audio and
publication approval are separate from this visual review.

Prepared 2 November 2026. **This is a checklist, not a completed delivery.**
The H3 reference technical run and protected candidate-instance deployment have
closed. Revised motion runs and their output acceptance remain open. Check the current progress report before starting
inference. Run large models on Linux; Windows supplies the browser and source
checks. Keep the original first-milestone LTX/Wan evidence.

Both Linux instances now serve the tested `579bd123` / `d357832a` application
pair. The revised Pika camera/action workflow has passed actual browser Run/Stop
at evaluation 1 of 6, zero cancelled outputs and worker recovery. Its fresh
124-frame clip and full muted playback pass, with subject/camera movement
observed and tighter ending framing recorded. An unchanged repeat reuses all
five nodes; scoped release preserves both outputs. The eight-second two-scene
clip also passes all 192-frame/media/playback checks, with a native cut near
4.17 seconds and movement in both shots. Its plaza camera is steadier than the
requested arc. Both clips are now available for private review. The revised
reference-motion clip now passes all technical/full muted playback checks,
retained identity and useful boat/camera motion. It is first on the private
review page. Wan image-to-video also passes all 121 frames and full playback,
with recognizable opening identity and useful movement; its strong camera
approach heavily crops the ending sail. Both new results are available and
the model queue/caches are idle. Full creative/audio approval remains open.

The latest user review accepts the four H3/Pika clips' image quality but finds
their motion insufficient. The old unstarted reference/Wan schedule is held.
Eight revised saved workflows use **My workflows → Video motion**; all eight
already pass real built-client Save/reload and exact executable-graph checks.
Three Pika motion results are now available for private review: camera/action,
two scenes with a native cut, and reference-driven boat motion. Their individual
framing/camera limitations and human listening remain open. Wan image also
has a completed native result with its ending-crop limitation recorded; the remaining H3 and first-frame motion variants are unrun. The original **Video candidate**
workflows below remain available as baselines, not improved-motion evidence.

Start the next quality review with **Pika — Camera tracking and cyclist action**:
expect visible pedalling/wheel rotation, lateral camera travel, changing view
angle and foreground/background parallax. Then review **Pika — Two scenes with
a timed cut**: expect eight seconds at 24 FPS, a forest-to-plaza cut near four
seconds, the same cyclist/bicycle, camera travel in both locations and matching
sound changes. First-frame variants should preserve the opening boat identity
but allow subsequent travel, heel, wake and a camera arc; Last Frame is omitted.
Reference variants preserve identity without fixing the opening composition.
See [the versioned recipes](motion-recipes.v2.json) for exact prompts/settings.

Review full playback, temporal consistency and original audio. A playing file,
distinct frames or successful encoding alone cannot pass the motion brief.
Keep failures as separate versions. Do not queue these while the H3 baseline
or another model task is active. Pika's longer duration requires separate
resource admission; no unsupported sampler override is part of these recipes.

## Find the workflows

Open **My workflows** in the updated candidate app. Seven saved candidates use
the prefix **Video candidate**:

| Family       | Candidates                                                | Frozen generation settings                                    |
| ------------ | --------------------------------------------------------- | ------------------------------------------------------------- |
| MiniMax H3   | Text and audio; First and last frame; Reference and audio | 960×544, 124 frames, 24 FPS, 50 sigma points / 49 evaluations |
| Pika H3 Flex | Text and audio; First and last frame; Reference and audio | 832×480, 124 frames, 24 FPS, fixed six-evaluation sampler     |
| Wan 5B       | Opening image                                             | 512×320, 121 frames, 24 FPS, 50 steps, guidance 5             |

Model Setup, Encode Inputs, Denoise and Video Output remain the main stages.
Reference assembly and input media are separate when needed; four is not a
stage-count limit. The saved inputs and prompts are included locally. The
required Pika adapter is part of its selected task, not a style switch.

The seven candidates are ordinary saved workflows, with separate immutable
native recipes and private evidence. They are not public Gallery qualifications.
The five existing LTX/Wan functional workflows and their limits are described in
[the first milestone](FIRST-MILESTONE.md).

## Quick editor checks

1. Open each candidate. Check the correct model, task and source media. H3/Pika
   should use the tested Custom/Group Disk settings; Wan image uses the tested
   resident configuration. Do not run these on the Windows GPU.
2. Inspect the meaningful stages and their internals. Pika must not offer CFG,
   negative prompt or an editable Steps control. Its Video Output joins actual
   video and generated audio. Wan retains its declared sampling controls.
3. Save, switch tabs, reload and reopen. Model, mode, prompt, seed, dimensions,
   input order, wires and layout must remain. Test an explicit stage separation
   and Undo on a copy; opening a document must not separate or regroup it.
4. Change a model/task on a copy. Compatible inputs should remain; incompatible
   ports or fixed-sampler controls need an explicit review. No download or model
   execution should start merely from browsing, importing or changing a task.

The built-client check already exercised ordinary Save/reload for all seven.
Its saved documents are also compared with the frozen executable graphs. This
does not replace native Run/Stop or media acceptance.

## Generation and media checks

Wait until the validation queue is idle. Keep the tested default settings for
quality review. Actual observed H3 text/keyframe runs took about 2¼ hours each;
reference sampling is substantially slower, currently estimated at 6–6½ hours
on this Linux machine. Pika text/keyframes took about 26–29 minutes. The revised Pika
reference case took 32 minutes 23 seconds; Wan image took 10 minutes 24 seconds. These are not Windows performance claims.

For a 2–3 hour manual session, review the retained full-quality clips and use a
separate engineering copy for a fresh H3 interaction check. Its previously used
512×288, 124-frame, two-sigma-point/one-evaluation preset took about nine minutes
and failed the quality brief. Label it as an engineering probe; do not replace
the full recipe or accept that output as the quality example. Pika's six-step
sampler remains fixed.

- Check the exact new task and Video Output rather than a source-media preview
  or an old cached image. Play the whole clip and listen to its original audio.
- Confirm the reference boat/keyframes remain recognizable, motion is coherent,
  and there are no black frames, severe flicker or frozen stretches. Review the
  first, middle and final moments and audio alignment.
- Repeat an unchanged successful engineering run. The executor should reuse
  eligible stages and retain the correct task/output ownership. Change the seed
  and prompt separately and check fresh computation where required.
- Stop a separate engineering run after real progress. It should settle as
  cancelled, leave the queue usable, and avoid reporting a partial clip as a
  completed output. Check a subsequent bounded run.
- Release only the owning Model Setup/cache. Completed clips and saved media
  must remain playable. Do not delete model files during this test.
- Export Service/API from a copy. Confirm named inputs/outputs and exact model
  pins; runtime preparation and metadata export are separate from executing its
  CLI. Do not claim a model-backed service run from inspection alone.

## Boundaries

Auto is not qualified for H3/Pika here. Other hardware, larger geometries,
additional offload combinations and arbitrary video adapters are not covered by
these recipes. Model tests use the accepted non-commercial license terms. No
video publication is approved. The whole-app Gallery gate also remains held;
local candidate readiness does not close that separate release gate.

Preserve original failed attempts and diagnostic outputs. Record manual findings
with the candidate title, actual task ID, changed settings and observed result
using [the logging procedure](logging.md).
