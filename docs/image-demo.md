# Image demo checkpoint — 22 September 2026

Open MoDiff at `http://127.0.0.1:8088` on the qualified workstation. In the **Workflows** sidebar, under **My workflows**, search for `Demo -`. Use the saved workflow settings; the Qwen developer starter otherwise defaults to a larger 2048-pixel output.

| Saved workflow                 | Purpose                                                        | Settings            |
| ------------------------------ | -------------------------------------------------------------- | ------------------- |
| Demo - FLUX Schnell            | Fast generic image generation                                  | 1024×1024, 4 steps  |
| Demo - Qwen 2.1 Text           | Complex prompts, typography and native attention-context reuse | 1024×1024, 40 steps |
| Demo - Qwen 2.1 Reference Edit | Two uploaded reference images and an edit instruction          | 1024×1024, 40 steps |
| Demo - Saved Block             | Reinsert a saved generic image graph and edit its seed         | 1024×1024, 4 steps  |
| Demo - Modular Z-Image         | Visible native Modular Diffusers composition                   | 1024×1024, 8 steps  |

The named workflows and JSON backups are local demo artifacts. To recreate a graph,
choose Developer → New → Text to image (or Multiple reference images), select the
model, preview and create the workflow, select Custom resource policy, then use the
sizes and step counts above. Choose `ZImageModularPipeline` for the Modular example.
Upload your own two references for the Qwen edit; JSON backups do not embed those
files. The saved recipes use bfloat16 and model CPU offload.

## Suggested demonstration

1. Open FLUX Schnell, change the prompt or seed and Run. Repeat unchanged to show output reuse.
2. Switch Creator / Developer. The editable graph and resource policy persist; the starting dialog and authoring tools differ.
3. Open Modular Z-Image and inspect Load Models, Encode Prompt, Denoise and Decode. Change the prompt or seed and run again.
4. Open Qwen 2.1 Text. Inspect **Reuse attention context**. This controls native KV reuse within one generation; it is separate from MoDiff's reuse between graph runs.
5. Open Qwen 2.1 Reference Edit and inspect the Load Image connection. Keep the provided references for a predictable demonstration.
6. In Developer, use Export → Service package to expose prompt and seed, and name the preview output. For reference editing, also name the Load Image file input so callers can supply both references. The exported package runs through MoDiff's queue and runtime; it is not standalone Diffusers Python.

## Qualification limits

This checkpoint covers representative image demonstrations on AMD Radeon 8060S / ROCm 7.2 with approximately 121 GiB shared system RAM. It does not prove Windows 16 GiB VRAM / 32 GiB RAM viability. Qwen 2.1 uses standard Diffusers: the reviewed upstream snapshot has no native Qwen 2.1 Modular pipeline.

The exhaustive W8/W9 image/audio campaign remains open. Audio, video and unsupported optional-runtime targets are not newly qualified by this image demo. Qwen weights remain subject to the publisher's research license.

## Checks completed for the implementation

- Backend base gate: `./scripts/with-runtime-env.sh .venv/bin/python -m pytest -q` — 3,538 passed, 525 skipped, 10,109 subtests passed.
- Reviewed optional runtime gate: `./scripts/with-runtime-env.sh .venv/bin/python scripts/test_reviewed_optional_runtime.py -q tests` — 4,101 passed, 21 skipped, 10,820 subtests passed. These overlap with the base gate; do not add the counts together.
- `uvx --from ruff==0.12.7 ruff check . --select E9,F` passed.
- Client `npm run check` passed, including types, unit tests, production build and bundle checks.
- Full mocked browser run: 239 passed, 3 failed. After excluding generated review evidence from Vite's file watcher, all three exact failed cases passed in an isolated rerun. The full 242-case suite was not repeated after that configuration-only fix.
- Production frontend bytes match the tested client build. Preservation checks cover 112 original model snapshots and 1,696 files: targets, sizes, modification times and recorded metadata hashes are unchanged; operator approvals are byte-identical. This is not a fresh hash of every weight payload.

## AMD reference-edit fix

On this ROCm stack, SDPA in Qwen's vision encoder produced non-finite prompt embeddings and a black/transparent output. The shared runtime now selects Transformers' public eager-attention implementation for SDPA vision subconfigs on ROCm. Text and Diffusers denoiser attention retain their existing implementation; non-ROCm hosts and other explicit vision implementations are unchanged. The full two-reference UI edit now yields a valid image. This does not add a model-specific node or frontend branch.

## Live evidence and scope

FLUX Schnell passed eight UI/service cases, including reopening and prompt/seed
changes. Qwen 2.1 text generation passed ten cases, including native KV on/off,
forced recomputation, service export and Creator/Developer invariance. After the
ROCm fix, the two-reference edit passed UI baseline, cache-off, forced cache-on
and reopening. That campaign exposed a service scalar/list mismatch, which was
fixed; its separate eight-case recovery passed multi-file service execution,
repeat reuse, seed/prompt edits, UI/service agreement and workspace invariance.
Modular Z-Image passed seven UI/API/CLI service cases. The Saved Block was created
from a real graph, reinserted, seed-edited, executed, saved and restored.

For two 1024-pixel references and 40 steps, a warm Qwen edit took 598.26 seconds
with native KV reuse off and 280.36 seconds with it on: 2.13× whole-task speedup
in this single paired observation on this AMD host. Warm cache-on output matched
the original byte-for-byte. The text-only comparison was 181.10 versus 172.14
seconds (1.05×); its run overlapped mocked browser checks. Neither result is a
claim about the publisher's A100 benchmark or general speedup across prompts.

The final executable backend source was tested at `7d412c6`; the client was
`03d4c22`. Earlier image/cache observations used `9cc1ba1`/`267683b` and
`433a19e`/`03d4c22`. The subsequent changes fix ROCm vision attention, development
file watching and multi-file service validation; the final reference baseline
reproduced the earlier fixed output byte-for-byte. Documentation commits do not
change the tested production bundle or executable code. Export service packages
from the final running app, since their requirements include the backend revision.

Detailed local receipts, original failures, recovery evidence and JSON workflow
backups remain under `reviews/creator-developer-w9`, `reviews/image-demo` and
`reviews/creator-developer-w10-upgrade`; they are not shipped as public Gallery
assets. This is a demo checkpoint, not completion of W8, W9, W10 or H1.
