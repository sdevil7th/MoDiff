# Image and cross-platform validation

Use this procedure to qualify image nodes, workflows and templates on an exact
backend/client pair. It complements [the engineering procedure](cluster-engineering-lessons.md),
[contributor gates](../CONTRIBUTING.md#validation), [image template validation](image-template-validation.md)
and [accelerator setup](accelerator-installation.md). A Git checkpoint or a
successful image does not establish every platform, resource recipe or UI path.

## Freeze the inputs

Record both Git commit IDs and any dirty diff, the served client asset hashes,
Python and package versions, exact Diffusers distribution/source identity, OS,
GPU architecture, driver/runtime, physical RAM, dedicated or shared GPU memory,
swap/pagefile, and available model/offload disk space. Keep credentials and local
paths in private evidence, never in public receipts.

Keep the existing reviewed ROCm stack. A generic CPU/CUDA `uv sync` can replace
it. Use `scripts/with-runtime-env.sh` where the installed AMD profile requires
it. Discover the actual device and supported profile before installing anything.
An AMD GPU exposed through `torch.cuda` is still a HIP/ROCm target; a CUDA API
name does not establish NVIDIA kernel compatibility.

Ordinary released-runtime and explicit FLUX3 evaluation runs are separate lanes.
Follow [the exact evaluation artifact contract](accelerator-installation.md#explicit-diffusers-source-evaluation).
The private universal wheel is not supplied by Git. Transfer that exact artifact
separately and verify its hash; do not copy a Windows environment, package overlay
or CUDA lock onto Linux. A locally rebuilt wheel with different bytes needs its
own review, even when its version and source revision look the same.

Preserve workflows, Saved Blocks, history, inputs and existing generated output.
Use an isolated data directory and loopback backend; SSH access uses a tunnel.
Run one heavy model workload per machine. Parallelize across machines, but keep
RAM-heavy full test suites out of measured inference windows on each host.

## Inventory and proof ledger

Enumerate the current public image templates, canonical image workflows,
execution profiles/tasks, private authoring drafts and deterministic image nodes
from the maintained inventories. These lists overlap and must not be summed.
For every exact route or template, retain separate states for:

| Proof | Required record |
| --- | --- |
| Static contract | Exact profile/task/revision, graph, fields, selected components, original defaults and declared eligibility |
| Software | Focused regressions, installed-library API checks, source-catalog checks, complete gates, actual skips |
| Native UI | Real insertion/edit/run gestures on the served bundle, screenshots, submitted graph and visible output |
| Trained execution | Task ID, immutable artifacts and LoRAs, requested/consumed controls, output dimensions and hashes |
| Repeatability | Cold inference, cache-hit dispatch and independent warm recomputation distinguished |
| Quality | Prompt adherence, edit/reference preservation, mask/seam/detail checks and adapter effect |
| Resources | Actual free/allocated/reserved memory, host RAM, commit/swap, time, release and next-run observations |
| Eligibility | Auto, template, hardware and publication acceptance recorded independently |

Use `passed`, `failed`, `untested`, `missing artifact/access`, `capacity blocked`
or a concrete runtime restriction. Preserve failed receipts. A later focused
repair does not turn the original failed whole campaign into a pass.

## Software and basic application gates

Before freezing a cross-machine checkpoint, compare working-tree bytes with the
Git index. Apply the repository's declared line endings before calculating raw
file hashes. Regenerate affected hash bindings and verify the exact bytes that a
fresh checkout will receive; a passing Windows working copy alone is insufficient.

1. Run all commands in `CONTRIBUTING.md`, including package compatibility and
   ordinary preflight. Supply the immutable catalog fixture for source checks.
   Run the compatible client `npm ci`, `npm run check` and relevant maintained
   browser suites. Baseline skips do not prove missing FLUX3 APIs work.
2. Run the explicit installed evaluation lane and its no-download component,
   factory, scheduler, LoRA, dtype and offload tests. Keep source-audit imports
   distinct from a genuinely installed evaluation distribution.
3. Compile every image canonical graph and classify every route. Check all
   template defaults, required inputs, loader identities, adapter connections,
   preview owners, export parity and unchanged saved graphs.
4. Rebuild and mirror the client, preserving `web/user`. Regenerate affected
   inventories in dependency order: upstream coverage, template candidates,
   template authoring, Comfy contract/evidence. Review semantic changes separately
   from byte hashes. Verify a fresh server serves the exact shell and assets.
5. Before expanding the model campaign, run one established small image workflow,
   then original-settings FLUX3 Distilled and Base through the ordinary UI.
   Exercise optional prompt enhancement and owner turnover early, rather than
   leaving the first complete user journey until the end.

Also smoke ordinary setup/upgrade/start/stop, settings persistence, registry and
model browsing, downloads with retry/Repair/offline behavior, file import and
export, history, Saved Blocks, workflow tabs, and keyboard/Undo interactions.
Check missing optional runtimes and custom extensions through their existing
explicit install/enable boundary. Include representative existing audio and
video workflows where assets and capacity permit, so shared graph, queue and
resource changes receive cross-modality regression coverage. Record unavailable
models and unsupported platforms explicitly rather than declaring the whole app
qualified from image-only results.

## FLUX3 finite matrix

Classify all 48 combinations independently on every host:

```text
{Base, Distilled}
  x {dense BF16, official FP8r, official NVFP4 fallback}
  x {plain, compatible trained LoRA}
  x {resident, model CPU, group CPU, group disk}
```

Execute every eligible tuple. For E eligible tuples, the minimum is 3E graph
dispatches: E cold inferences, E unchanged cache controls, and E independent warm
inferences. Use fresh instances or explicit output-cache release to prove actual
recomputation. Keep the original 1024-square recipes: Distilled 16 steps and Base
50 steps with real CFG 3.5. Resource comparisons must be explicit user-visible
instance changes; never silently shrink dimensions, steps or precision.

The exact Windows Torch rowwise FP8r restriction is not a blanket FP8 hardware
restriction and does not transfer to ROCm. A separate observed Linux restriction
refuses FP8r before weight allocation and old-owner cleanup on the exact
Torch `2.9.1+rocm7.2.0.git7e1940d4` source build
`7e1940d4b11fd6128be4c42ba41567ec5ab87102`, HIP `7.2.26015-fc0010cf6a`,
`gfx1151` device with capability `(11, 5)`. The official BFL FP8r linear selects
`torch._scaled_mm`, whose installed implementation requires ROCm MI300+.
This is a separate genuine tiny-kernel failure, not a blanket AMD refusal or a
claim about other builds/devices. The same tiny probe produced finite NVFP4
fallback output; that does not prove trained-model output or memory fit. No
format fallback or saved-setting rewrite is performed. Test genuine quantized
kernels before heavy allocation. Fused NVFP4 on Blackwell is a separate lane from fallback.
Shared-memory AMD systems must account for CPU and GPU pressure together; CPU
offload is not automatically lower-total-memory execution.

Require one viable original-settings Auto route per variant, then separate
evidence for each further Auto tuple intended to be enabled. Current private
PUP Auto contracts can be platform-specific. An intentional refusal on another
platform must preserve the outgoing owner and explain the missing qualification.
Do not bypass the boundary or silently run Expert to manufacture an Auto pass.

## Conditioning, stages and adapters

Use exhaustive cheap validation tests plus trained cases for each distinct
consumer and ownership transition:

- Text-to-image and reference editing for both variants; empty, Unicode, long
  and negative prompts where supported; raw/structured conflict refusal.
- Reference counts 0, 1, 2, 8, 9 and 10, plus rejection above the declared limit;
  reorder, replace, crop, resize and explicit rebind. Edit requires a real source.
- Layout placement, relocation and removal; source/target identity, finite and
  ordered coordinates, duplicate IDs, zero and maximum region counts, stale
  geometry and missing reference rejection. Inspect actual arrangement quality.
- Declared output-count, dimension/aspect and sequence-length limits, batch
  expansion once, actual FP32 VAE decoding and direct buffer reads.
- Change prompt, seed, reference, guider, scheduler and decode settings one at a
  time. Assert which stages recompute and reject stale material or foreign owners.
- LoRA add, scale effect, zero, disable/re-enable, replace/remove, ordered two-
  adapter chains and independent return to baseline. Verify real trained effect,
  not just successful injection. Reject wrong targets, ranks, shapes, metadata
  and foreign owners before mutation; test failure/cancellation rollback.
- Base/Distilled/FLUX.2 switching with compatible value preservation; inspect
  unsupported control removal, Cancel, Undo/Redo, Create Block, saved reinsertion,
  refresh and A-to-B-to-A model/cache behavior.

## Every image template and utility

For each public image template on every eligible host, verify native creation,
unchanged creator defaults, readiness, cold generation, changed-seed/prompt
inference, cache reuse, save/reopen/export and exact-task preview persistence.
Missing models or insufficient capacity are explicit open rows. A template
construction test is not trained-output acceptance.

Prioritize migrated Fill/Canny/Depth/Redux stages, masks/outpaint, Qwen reference
and layered outputs, FLUX Dev/Krea/Kontext and LoRAs, Z-Image guidance, then the
remaining catalog. Test the exact task: some outpaint presentations execute an
inpaint recipe with a prepared canvas/mask. Inspect every generated border and
seam, not only global image contrast.

Cover load/save, resize/crop/pad, RGB/RGBA conversion, masks/inversion/blur,
composite, Canny/depth, grids/batches, upscaling and previews. Use deterministic
fixtures for EXIF rotation, alpha, grayscale, odd dimensions, batch order,
invalid/corrupt/oversized files and path containment. Assert exact pixels or
documented numeric tolerances where deterministic. Learned depth/upscale nodes
also need real pinned-model output and visual assessment.

Six FLUX3 Gallery candidates remain a separate deliverable: Distilled T2I,
Base T2I, Distilled reference edit, Base reference edit, trained LoRA, and optional
PUP layout. Do not publish a layout template while structured PUP output is
unfinished. Generate via maintained tooling; keep media outside Git and obtain
separate publication authority where required.

## Prompt enhancement and recovery

Test disabled PUP first: the authored prompt must work without loading it.
Then use real Preview, Regenerate, Apply, Disable, one-step Undo and reload.
Bind responses to current workflow/model/prompt/reference/layout; stale responses
must refuse. Confirm actual submitted and consumed enhanced text and inspect the
result. Image-to-PUP-to-image turnover with the original prompt is useful owner
proof, but does not establish Apply or enhanced-image quality. Structured layout
enhancement needs valid IDs/counts/geometry and atomic caption/layout Apply.

Test active and queued Stop, Stop-to-Retry, cancellation during download,
verification, loading and adapter mutation, five active refresh/reconnect cycles,
editing a different tab, controlled worker replacement and unchanged-settings
recovery. Preserve previous history and outputs, and verify declared output URLs
and downloaded bytes against the correct task/hash. Poll lightweight status
during inference and measure control/metadata responsiveness.

Check missing/corrupt shards, Repair, offline restart, exact revision/hash drift,
symlink/reparse/traversal refusal and remote-code boundaries. Browsing, discovery
and Auto inspection must neither download nor install nor execute custom code.
Check fresh per-owner disk leases, repeated release, configured temp roots,
cancelled/forced-worker exits and orphan recovery. Retain independent outputs
after owner teardown and inspect memory before the next allocation.

## Parallel work and closure

Use one shared task ledger and separate branch/file ownership. Windows can focus
on WDDM/commit, offload, PUP UI and recovery; ROCm can focus on actual HIP kernels,
shared-memory pressure, larger dense recipes and independent template regression.
Both hosts still run core FLUX3 acceptance; another host's pass cannot qualify it.

Compare exact recipes, ordered media and semantic quality across platforms.
Require same-host upstream/stage pixel parity where the contract supports it;
do not assume identical CUDA/ROCm pixels. Freeze final source, rebuild, run final
software gates and repeat affected critical trained journeys. Report every
remaining failed, blocked and untested row before enabling support claims.
