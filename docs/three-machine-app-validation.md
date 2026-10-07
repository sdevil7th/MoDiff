# Three-machine application validation

Use this procedure on the existing local machine, a dedicated AMD Linux cloud
machine, and a Windows machine. Test the same reviewed backend/client commits
and original workflows, while keeping each machine's runtime and results
separate. A successful local image campaign or CPU CI run does not qualify a
different accelerator, an untested media family, or every advertised route.

Follow the [engineering procedure](cluster-engineering-lessons.md),
[workbench acceptance](workbench-acceptance.md), and
[runtime support matrix](runtime-support-matrix.md). This guide coordinates
their checks; it adds no executor, installer, or qualification framework.

## Prepare source and isolate the run

Finish the local implementation, code review, and applicable contributor gates
before preparing the paired commits for the other machines. Record both commit
IDs, lockfile hashes, and the validated production asset hashes. Use the bundled
production app for generation evidence. Test Vite reconnect behavior separately
with its client source identity recorded; changing HMR source during inference
does not produce one immutable application result.

Use a fresh test checkout and a private run directory on each machine. Keep
credentials, custom-code approvals, installation journals, personal workflows,
virtual environments, generated media, and operator caches out of source
transfer. Keep required curated files under `data/`; they are application
contracts. Recreate the environment on the destination, rather than copying
another platform's virtualenv. Preserve the original input bytes and their
ordered bindings when relocating files.

Set all mutable configuration paths (`work_dir`, `data`, `images`, `videos`,
`audio`, `models`, `upscalers`, and `temp`) under the private run directory.
`app_root` remains the checkout. Use an explicit test Hub cache and record
whether the run is online or offline. Existing model access, licenses, and
custom-code consent remain prerequisites; a portable workflow does not grant
them. Do not change operator installations, clear their caches, or stop their
workers to obtain a clean test.

| Machine | Setup and first checkpoint | Platform boundary |
| --- | --- | --- |
| Existing local machine | Use its reviewed installed profile and `scripts/with-runtime-env.sh` on Linux. Verify the actual device, source, private storage, and idle owned queue. | Shared-memory or integrated-GPU observations do not establish dedicated-VRAM fit. Reuse earlier evidence with its original source/runtime scope. |
| Dedicated AMD Linux cloud | Follow [AMD cloud preparation](amd-cloud-qualification.md) and `scripts/prepare_amd_image_validation.py`. Identify the actual GPU/partition, architecture, visible device count, memory, host RAM, storage, and driver/runtime before allocating models. | The existing Instinct preview targets `gfx942`; MI100/`gfx908` needs a separate reviewed profile. The AMD preparation helper is Linux-only and model-free. |
| Windows | Use the [destination installer](accelerator-installation.md) or native `uv` commands with the actual selected accelerator extra. Check preflight before launching the supervised app. | Ordinary NVIDIA generation uses eager SDPA and the default allocator. Optional compilation, Intel XPU, and AMD Windows have distinct readiness boundaries; Linux results cannot qualify them. |

For a fresh NVIDIA Windows checkout, run from the backend in PowerShell:

```powershell
uv sync --extra cuda
uv run --extra cuda python -m modiff.preflight --json --check-port 8088 --fail-on-error
uv run --extra cuda python main.py
```

CPU setup uses `--extra cpu`; other platforms follow
[developer setup](developer-setup.md). Do not run generic CPU/CUDA sync over an
existing specialized ROCm environment. Keep the server on loopback; use an SSH
tunnel for the cloud browser. Verify the actual worker listener PID, creation
time, working directory, `/health` instance/startup identity and served assets.
A shell, supervisor, or Windows venv redirector PID is not the model worker.

## Checks that can finish without production models

Run the applicable [contributor validation commands](../CONTRIBUTING.md#validation)
in a separate test environment. Preserve failed and skipped tests. Full backend
pytest isolates extension discovery; private subprocess probes must also isolate
configuration and extension loading before importing the registry. Registry
inspection must not execute the operator's custom Python.

Existing model-free runtime and HTTP checks can be run from the backend:

```sh
./scripts/with-runtime-env.sh .venv/bin/python scripts/smoke_base_runtime.py
./scripts/with-runtime-env.sh .venv/bin/python scripts/smoke_service_package.py
```

On Windows, invoke `.venv/Scripts/python.exe` directly with those script paths.
The first uses tiny randomly initialized CPU components; the second uses
temporary storage and the ordinary queue/service path. Neither downloads weights
or proves full-model output. The HTTP smoke binds its own temporary port; do not
point it at an operator's running server. Client `npm run check` and
`npm run check:ui` supply their unit/build and browser evidence separately.

In the real production browser, check these model-free journeys on every machine:

- Create a text/value graph, connect it to a preview or export, Run once, and
  match the visible output to the completed task receipt. Inspect the same task
  in Queue; refreshing must retain one task rather than creating a duplicate.
- Move and resize ordinary nodes; connect/reconnect compatible ports, reject
  incompatible ports and cycles, and exercise Undo/Redo. Save, reload the owning
  document, reopen it, and compare its effective API graph and authored values.
- Create a supported model/task graph without running it. Inspect real stages,
  required loaders, model choices, controls, grouping/expansion, and interface
  edits. Model or task switching must retain compatible authored values and
  report incompatible inputs without silently discarding them.
- In the private test checkout only, review and intentionally add the repository
  [PromptTools custom example](custom-nodes.md#a-small-ordinary-node). Run its
  text graph, edit its installed source, Reload at idle, and verify a fresh
  result plus cache invalidation. Test disable/re-enable and an import error.
  Listing sources or opening a workflow must not authorize execution. Never
  include approval files in a transferred package.
- Load approved small image, audio and video files. Verify input previews,
  typed processing connections, output preview/export and decoded metadata.
  [LightPaletteDirector and AudioEnvelope](custom-nodes.md) exercise custom
  decoded-media processing without model loading. This is media I/O evidence,
  separate from generated media quality.
- Export an edited text graph as a service package and invoke it through the
  [existing service runner](service-prototyping.md). Retain named input/output
  and runtime-manifest checks; do not generate a second inference script.

Record startup/health, real HTTP responses, browser console exceptions, visible
graph/queue screenshots, saved workflow, effective API graph and output bytes.
Mocked browser coverage stays labelled as mocked even when all checks pass.

## Full-model sequence on each admitted machine

Choose exact routes from the current `/model_capabilities`, operation starters,
and execution specifications. Freeze the selected route/template list before
running. List unavailable weights, access, optional runtime, unsupported device,
or resource-plan failures explicitly. A family name or a model picker entry is
insufficient execution authority.

Run one workflow at a time on one owned GPU. Keep original resolution, steps,
seed, batch, negative text, guidance, strength, references, LoRAs, scheduler,
attention and VAE policy. Set a per-run wall-clock bound and an ordinary Stop
plan based on the actual recipe and available test window. If the bound expires,
inspect the task and cancel it through the app when required; wait for terminal
state before retrying. Never resubmit an unknown active task or shorten an
original recipe to claim completion. Record timeouts and deliberate cancellation.

Start with an admitted full canonical short-step image recipe and follow this
sequence without model release between the first four generations:

1. Cold ordinary Auto Run. Capture the accepted resource plan, requested and
   effective placement, actual model/artifact identities and every output.
2. Same-input warm Run. Prove new model forwards or actual task execution,
   rather than a persisted preview or reused generation node result.
3. Seed-only Run, then positive-prompt-and-seed Run with the original negative
   text retained. Verify re-encoding after a completed guidance loop and new
   output/noise lineage where observable.
4. Refresh the owning page during a separate active run. Retain local workflow
   state and the same task/progress/output across reconnect. Perform a separate
   Stop, wait for terminal cancellation, then retry the full recipe through the
   ordinary Run action. Only stop the task owned by this test.
5. At known idle, release the test's model/cache owners through the app and
   select another model. Verify queue responsiveness and the next generation.
   Record failures and the [recovery outcome](image-runtime-recovery.md).

Then batch full recipes by cached model family. Exercise resident `none` and
reviewed CPU/disk offload in separately labelled Custom runs where admitted.
Auto may choose a different effective policy from the creator or predecessor;
retain that outcome and use an explicit matched-policy run for strict recipe
comparison. Equal offload strings do not prove identical physical residency.
Do not bypass Auto rejection with an unlabelled Custom fallback.

| App scope | Full-model checks after prerequisites pass | Review outputs as |
| --- | --- | --- |
| Images | Z, Qwen T2I/Edit/Plus/Layered/Control, Flux native families, and reviewed Fill/control/Redux routes; single/multi-reference, mask/outpaint, named LoRA and upscale branches. | All Preview sinks and local item indices, geometry, source/reference retention, mask/seam behavior, requested edits/text, and full images. Upscale retains both generated and finished outputs; Layered retains every ordered RGBA layer. |
| Audio and speech | Current admitted generation/variation/edit/continuation routes and transcription, using exact rate/channel/duration and task-specific inputs. | Decode and playback, audible content, lyrics/transcript, clipping/silence, edit boundaries and actual duration. Frame-quantized output is not padded to claim an exact requested duration. |
| Video | Current admitted text/image/video-conditioned routes, with exact legal frame counts, FPS, input order and runtime dependencies. | Decode/frame count/duration, playback, non-black finite frames, motion and reference/continuation consistency. A short smoke does not qualify [long-video continuation](long-video-qualification.md). |
| Text, vision, depth and other advertised tasks | Exact compatible existing operations with immutable artifacts and declared inputs; 3D or multimodal routes only when their current contracts and prerequisites admit execution. | Actual text/task fidelity, depth orientation/geometry, media order, or valid rendered/exported assets. Palette discovery alone does not qualify these tasks. |

Full execution and aesthetic/listening review are separate outcomes. Custom
decoded-media nodes can process compatible family outputs; they do not make
latent or component formats interchangeable. Larger or guarded model routes
follow [large image-model validation](large-image-model-validation.md), with
their access/runtime gates intact.

## Evidence, comparison and finish

Keep one private evidence row per source/machine/route/recipe and attempt. Include:

- Paired commits, dirty-source identity if applicable, lockfiles, pinned upstream
  commit, served bundle/public contract hashes, instance and actual worker
  PID/creation time/cwd; driver, Python, Torch/HIP/CUDA, Diffusers, Transformers,
  PEFT, OS, actual devices, dedicated/shared memory, host RAM and storage.
- Original visible and effective API graph, requested fields, ordered original
  input byte hashes and decode geometry, immutable base/auxiliary/adapter pins,
  actual consumed values and observed model/component/cache identities.
- UTC submit/start/progress/terminal/cleanup markers, task/run receipts, raw
  outputs by sink/index, decoded dimensions/count or rate/channels/frame count,
  original byte hashes, independent review and retained failures.
- Auto plans and graph allocator observations separately from OS sampling.
  Start `scripts/capture_runtime_memory.py --pid ACTUAL_WORKER_PID --output NEW_FILE`
  before allocation. Its maximum window is one hour; confirm each header/first
  sample and rotate one writer at known idle, retaining footers and sampling
  gaps. Samples are lower bounds, not memory budgets or VRAM-fit qualification.

Use [image-template comparison](image-template-validation.md) and the existing
`scripts/compare_image_template_outputs.py` with independently derived consumed
contracts and all ordered raw outputs. Compare baseline/current on the same
machine and runtime; cross-machine hashes are observations rather than an exact
numerical baseline. Keep source-specific historical passes separate from fresh
runs. Missing internal observations remain unavailable, not reconstructed from
the opposite result.

At completion, verify known terminal/empty queue, release only owned runtime
resources and stop only owned processes. Preserve the natural sampler footer,
before/after source/operator/model metadata and cleanup outcomes. Back up and
verify evidence bytes before separately authorized cloud teardown. Report
contract, browser, execution, repeat/recovery, quality, Auto/resource, and
platform results separately, with an explicit blocked/unrun list.
