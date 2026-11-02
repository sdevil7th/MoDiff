# MiniMax, Pika and Wan image testing checklist

Prepared 2 November 2026. **This is a checklist, not a completed delivery.**
The H3 reference run, subsequent Pika reference/Wan image runs and final deployed
pair still need to close. Check the current progress report before starting
inference. Run large models on Linux; Windows supplies the browser and source
checks. Keep the original first-milestone LTX/Wan evidence.

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
on this Linux machine. Pika text/keyframes took about 26–29 minutes. Reference
and Wan image timing remains pending. These are not Windows performance claims.

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
