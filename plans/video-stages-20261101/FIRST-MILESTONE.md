# First LTX and Wan testing milestone

Completed on 1 November 2026 for **bounded private testing**. The four-family plan
remains in progress. This is not a whole-app release or public-template signoff.

The deployed application pair is backend
`569be22033af60b0d0fda731231696a394c9c6ff` and client
`79cf779693acc6c032619c8972b30c06d2c07fa1`. Later plan-only commits do not change
those tested application revisions. Product changes remain in the existing
[backend PR7](https://github.com/sdevil7th/MoDiff/pull/7) and
[client PR7](https://github.com/sdevil7th/MoDiff-client/pull/7). Windows was the
only source-authoring location; Linux received exact published revisions.

## Delivered behavior

- Selected LTX and Wan workflows expose shared Model Setup, Encode Inputs,
  Denoise and Video Output stages. Actual upstream encoding, sampling and VAE
  decoding run behind the existing executor. Stage count is not capped at four.
- Fresh video drafts add ordinary typed image, video or audio loaders for their
  declared required inputs. Missing or incompatible sources produce a notice.
  Existing saved graphs are not rewritten.
- Owned conditioning and latent bindings reject foreign, changed or stale
  runtime data. Unchanged runs reuse computed stages. A changed video source
  retained the model while recomputing encoding, sampling and decoding.
- New video outputs capture the actual task, prompt and final format. Optional
  `graphTasksVersion: 2` extends the existing receipt vocabulary without
  rewriting historical records or promoting qualification.
- The selected LTX distilled starter supplies actual guidance 1 and 8 steps in
  both parameter values and defaults. Encoder sequence length is normalized to
  the reviewed 128-token bound. Bounds alone were insufficient.
- Retained template factories keep their real upscale/soundtrack branches.
  Shared settings preserve managed-document selector metadata while fresh fields
  retain strict typed validation and a separate immutable revision.

## Native evidence and testing scope

| Route                | Actual output checked                          | Boundary                                                                                                                                              |
| -------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| LTX text-to-video    | 704x480,81 frames,16 FPS; also320x256,9 frames | Generation, file, playback and unchanged-repeat evidence. The larger output predates the metadata-only correction; its computation path is unchanged. |
| LTX image-to-video   | 320x256,9 frames,16 FPS                        | Fresh default-guidance run, actual input hash, playback and unchanged repeat.                                                                         |
| LTX video-to-video   | 320x192,17 frames,16 FPS                       | Actual video input and fresh downstream computation with model reuse. Correct managed output was played separately from the source preview.           |
| LTX two references   | 320x256,9 frames,16 FPS                        | Ordered input hashes, actual generation and exact task/output-preview match.                                                                          |
| Wan 5B text-to-video | 512x320,49 frames,24 FPS                       | Completed larger native recipe reused. Every literal setting and typed wire in the saved test workflow matches that record.                           |

These are bounded route checks, not acceptance of every advertised resolution,
offload combination or creative template. Wan's pinned direct `WanPipeline` path
is text-only; its TI2V repository name does not establish an admitted image route.

Thirteen accepted task attempts comprise eight fresh video completions, three
cached repeats, one failure before video output and one intentional cancellation.
Two fresh clips are retained as failed visual diagnostics: low-resolution Wan
produced abstract imagery, and the first video-to-video check used that bad source.
The corrected source was derived from the earlier good Wan clip. Failed recipes,
outputs and result logs remain available. There are no public video uploads or
publication approvals.

The native Stop check became cancelled with zero outputs; the supervisor returned
a ready worker on the same source. The click raced from sampling into decode, so
this does not prove sampler-callback-only interruption. Six saved clips remained
hash-exact afterward. Scoped model release retained durable media; independent
media-loader caches can remain without retaining model weights.

Stage separation preserved literal values and typed wiring while rebasing the
separated encoder's identity. One Undo and saved reopening restored the exact
executable graph. Explicit separation is not a qualification-preserving edit.

## Validation

Exact local commands, environments, source identities and original logs are in the
private execution ledger. The maintained commands were run using the existing
reviewed CPU environment and immutable catalog source; native model work stayed
on Linux/ROCm with the existing reviewed runtime.

| Check                                           | Result                                                                |
| ----------------------------------------------- | --------------------------------------------------------------------- |
| Backend `python -X utf8 -B -m pytest -q`        | 8,696 passed,749 skipped,13,084 subtests,30 warnings;1,550.66 seconds |
| Pinned Ruff 0.12.7 `ruff check . --select E9,F` | Passed                                                                |
| Client `npm run check`                          | Passed on the final application source                                |
| Focused video client suite                      | 19 passed                                                             |
| Related operation-authoring/media suite         | 576 passed                                                            |
| Final dist/committed bundle comparison          | 119 files exact;118 served assets verified at deployment              |
| Production launcher preflight                   | Passed; dependencies/runtime unchanged                                |

The broad browser diagnostic recorded312 passed / 3 failed and was not a final
stable-source gate. The managed soundtrack selector defect was corrected and its
isolated browser case passed. The Klein hydration timeout passed in isolation.
The controlled-image LoRA readiness-summary mismatch still fails and needs
diagnosis before whole-app release signoff. The earlier shared-control diagnostic
had14 passed / 1 cold-editor timeout; that case passed in isolation. These outcomes
are retained separately, not combined into a full UI-gate pass.

## Manual testing

Open My workflows and start with **LTX quick video test - Linux**. The saved
image-input, video-input and two-reference quick tests include their media. Use
**Wan text-to-video test - Linux** for Wan. Its512x320/49frame recipe is the
reviewed quality example; the first cold native run took about 13 minutes, mostly
decoding. Do not substitute the failed320x192/17frame diagnostic to save time.

Check Run, unchanged repeat, prompt/seed changes, Save/reopen, ordered media,
stage separation/Undo, Stop and scoped cache release. Select Video Output's clip
when a source loader also has a video preview. Larger creative recipes and public
Gallery acceptance remain separate from these short functional tests.

## Remaining priorities

1. Diagnose the controlled-image LoRA readiness-summary mismatch before a global
   release declaration; do not weaken qualification or rewrite its assertion
   merely to obtain a pass.
2. Finish V4 existing-video media-attachment/task-switch convenience, truthful
   adapter eligibility messaging and output-package filename labeling. Generic
   video-source/condition-strength summary accounting is incomplete; the current
   graph, source probes and task records support bounded acceptance only.
3. Review and integrate the exact new Diffusers runtime for MiniMax H3/Pika;
   perform staged-loading/resource admission before downloading or executing
   large H3 artifacts. SSD capacity alone is not proof that loading fits RAM.
4. Complete the remaining native/offload/lifecycle rows, retained finishing
   branches, twelve creative templates and exact-media publication approval.
5. Run the final full release gates after that source stabilizes. Missing hardware
   remains unverified. Linux success is not Windows or universal Auto support.

## Lessons for the next session

- Registry metadata is not executor dispatch proof: actual classes must resolve
  through the executor's `.main` entrypoint. Test construction there.
- Assert actual parameter values/defaults and consumed calls, not just schema
  bounds. A mandatory range does not replace an inherited value.
- Share authored settings without erasing recorded selector metadata or relaxing
  fresh-field validators. Check both old managed and new ordinary graphs.
- Wait for upload acknowledgment and verify settled paths/hashes before freezing
  a recipe. A chosen local file is not yet a completed upload.
- A submission timeout is not proof of failure. Check actual task/client identity
  and queue ownership before considering a retry.
- An input preview is not output proof. Bind playback to the completed task's
  managed output file and measured geometry.
- Low-resolution success can be visually unusable. Keep failed recipes and reuse
  good completed source media instead of repeating large runs without a reason.
- Keep full gates at stable source checkpoints. Preserve earlier failures and
  distinguish focused, segmented, mocked and native evidence.
- Use unique private browser output directories from the first invocation. One
  early shared-control trace was overwritten by a later default-directory cleanup;
  its failed result log remains, but the binary trace is an explicit retention gap.
- Use explicit UTF-8 for repository documentation on Windows. Keep raw evidence
  private. Active engineering time was not reliably metered; do not present wall
  time or test/download waits as measured implementation hours.
