# Video pipeline stages

Fresh selected LTX 0.9.8 distilled and Wan 2.2 5B text-to-video graphs use ordinary
**Model Setup → Encode Inputs → Denoise → Video Output** stages. These are real
registered nodes in the existing executor. Expanding a visual group exposes its
actual nodes and connections. Optional upscaling, separate soundtrack generation,
audio fitting and delivery remain visible operations when a template needs them.
Four stages is a starting shape, not a limit.

New H3 and Pika graphs share Model Setup, prompt encoding, video/audio denoising,
decoding and export operations. Keyframe/reference workflows add the real media
preparation and VAE encoding stages. They retain the selected model in their
backend-owned state; they do not create model-named client nodes. H3's new
`minimax-h3:official-provider:v1` profile requires the same exact reviewed video
provider as Pika. It exposes Steps; Pika retains its fixed sampler. Old H3 nodes,
all three original specification hashes and the immutable library definitions
are preserved. Retained source-era profile selections require an explicit rebind
before execution under a different provider; saved values are not migrated.

## Declared selections

| Execution profile       | Tasks exposed by separate stages                            |
| ----------------------- | ----------------------------------------------------------- |
| `ltx-video:direct`      | Text, opening image, ordered reference images, source video |
| `wan-22-ti2v-5b:direct` | Text-to-video                                               |

The reviewed Wan 5B route uses the upstream `WanPipeline`. Its public model name
does not establish an image-to-video contract. Other video routes keep their
existing whole or native Modular operations. Saved whole-pipeline graphs are not
automatically rewritten. Stage availability is an authoring declaration; it does
not imply hardware qualification, Auto support or successful model execution.

The reviewed H3/Pika schemas keep video and audio VAEs in float32 while the
default component precision remains bfloat16. The selected large components,
including `transformer_ref`, use the existing owned group-offload path; the
small audio VAE uses the reviewed resident placement described below. Pika's
exact source runtime has isolated package qualification, which does not establish
published-model fit or output quality. See the video plan's milestone report for
the bounded H3 engineering evidence and remaining acceptance.

Pika's source-generated workflow and 26-block companion contracts retain their
own immutable Diffusers revision. The historical workflow/block snapshots keep
their original identity. The companion validates its complete three-workflow
topology against the source-qualified workflow contract; it grants no execution
or hardware qualification. The public library collection uses schema 7 for
explicit source supplements; individual definitions retain schema 6 and their
own source revision. Single-source schema 6 collections remain readable, and
existing definition hashes remain unchanged. Both repositories must be updated
together for the new collection envelope.

The ordinary operation catalog and starter builder read those reviewed source
groups without importing optional pipeline classes. Pika's three task starters
use the shared joint video/audio stages, including its selected reference helper.
This authoring metadata grants neither runtime activation nor model fit. Runtime
readiness, executable Cluster admission and native acceptance remain separate
checks; the historical hierarchy must not be relabelled as Pika's newer source.

The read-only `/huggingface/modular-conditionals` response uses collection
schema 2: its historical tree remains at the original revision, with explicit
schema 1 source snapshots in `sourceSupplements`. The client selects one exact
revision before compiling or displaying a hierarchy. Duplicate revisions,
conflicting pipeline identities and recursively nested source collections are
rejected. Historical standalone schema 1 snapshots and their generator retain
their original contract. This hierarchy collection enables structural authoring;
it does not grant edited-tree execution or runtime/publication qualification.

Pika's explicit `pika-h3-flex:official-provider:v1` Expert profile binds all three
tasks to the exact optional video provider and immutable model artifact. Its
shared-stage specification omits sampler overrides. Group Disk is an explicit
implemented strategy, including the resident audio VAE; it is not an Auto or
hardware qualification. Automatic template acquisition remains closed until the
finite per-task file selection and native acceptance are joined. This profile
does not promote a public template, Gallery clip or executable Hub Cluster.

## Execution and ownership

- `LoadPipeline` retains the existing immutable-artifact, resource, offload and
  library boundaries. Only an exact declared profile and its catalog revision
  grant separate-stage ownership. Local or advisory selections keep the ordinary
  whole operation.
- `EncodeInputs` calls the official text encoder once and retains bounded, ordered
  source media. Source-media VAE preparation stays in the official conditioned
  sampler. Unchanged encoding can be reused; changing the prompt, task, guidance
  enablement, owner or recipe invalidates it.
- `Denoise` uses the existing backend adapter and the official pipeline with real
  embeddings and latent output. It preserves distilled timesteps, scheduler
  restoration, progress and cancellation. LTX retains fractional frame rates.
- `DecodeLatents` applies the reviewed official inverse normalization and VAE
  decode under inference mode. Its latent binding includes the exact owner,
  material identity, geometry and post-sampling generator state. Cross-owner,
  fabricated, mutated or stale bindings are rejected before decode.

These bindings are process-local data. They are neither executable graph
representations nor portable execution receipts. Saving a graph persists values
and connections; reopening after a restart recomputes unavailable runtime data.
Release model caches through the ordinary scoped controls.

The client uses the backend's exact selected starter and
`generic-video-stages:v1` authoring recipe. The retained creative settings and
ancillary branches remain intact. A decoder and file-backed exporter are grouped
only when their actual connections allow it. An intervening upscaler or independent
audio branch remains distinct.

## Validation boundaries

Export Video with Audio keeps generated intermediates in its own temporary
directory and preserves loaded audio inputs regardless of their filenames.
It replaces the destination only after muxing succeeds, cleans intermediates
on failure, and rejects a destination that aliases its audio input.

An omitted or empty optional-adapter list means Model Setup has no optional
adapters to load. Malformed descriptors still fail the ordinary file boundary.
Required model adapters, including Pika's task adapter, are handled separately.

H3 and Pika call their conditioner's nested model rather than its outer forward.
Their backend execution metadata selects leaf-level group hooks for that
conditioner, including incremental loading, so embeddings and other layers
load at the entrypoints actually called. Reuse requires the same declared
offload type; another owner's component is never silently reconfigured.

Their small float32 audio VAE explicitly opts out of group offloading in the
official library because of weight-normalization hook ordering. Source-declared
`component_offload_modes` keeps that component resident, while the large
conditioner and transformer retain the requested group strategy. Placement and
reuse record the audio VAE's actual resident mode. Incremental loading refuses
declared unsupported types without a reviewed placement before loading weights. The
actual component's opt-out is checked again before group hooks are installed.

Ordinary split video receipts record LTX condition strength separately from
denoise strength, using the actual official adapter call. Source video evidence
identifies the ordered frozen PIL pixel sequence and frame count from the
existing material stamps. Its versioned SHA-256 is not the original file hash;
paths and media bytes are not placed in history. Other supported frame material
retains its count with an explicitly unavailable pixel digest. Legacy output
history is not backfilled from the current form.

The shared joint video/audio backend adapters preserve the exact selected
MiniMax H3 or Pika H3 Flex pipeline in every process-local state. They delegate
media preparation, encoding, denoising and decoding to the selected official
blocks. Ordered references use the selected library-owned dataclasses and
retain decoded video frame rates and audio sample rates. Saved H3 actions
continue to resolve their original contracts.

New run history recognizes both saved H3 actions and the declared shared H3/Pika
routes. Shared routes require the exact pipeline, workflow, block selectors,
single Model Setup owner and complete state chain. This task label grants no
runtime, Auto or template qualification and does not backfill old receipts.

Their actual loop progress uses the existing node progress bridge: totals count
real model evaluations, and Stop raises after the current official loop step.
The temporary per-instance progress methods are restored on success, failure or
interruption; sampler math and the upstream block hierarchy remain unchanged.

Pika's package-owned sampler fixes the six-step schedules; hidden or explicit
Steps, CFG and negative-prompt overrides are rejected. Its task adapter is a
required part of the model, not an optional style control. Model Setup verifies
the cached immutable Safetensors file through the existing LoRA file boundary,
loads it into the task-selected transformer with the official Pika loader, and
activates it at scale one before group-offload hooks are installed. Optional
style adapters are not admitted on this route. Reused loaders and downstream
stages check a weak binding to the actual component and its active, unmerged
scale-one layers. Model Setup rejects connected component overrides and checks
exclusive registered ownership before the first adapter injection. Component
reuse includes the required adapter identity, so an unadapted component from
another owner cannot be changed in place. Reusing an already bound component
does not reinject or reactivate it. A serialized flag cannot substitute for
that binding. Adapter declarations and no-weight tests alone do not enable a
public Pika template or establish model fit; exact model loading and native
acceptance remain required.

Pika's pinned index declares `Qwen2TokenizerFast`, while the reviewed official
blocks declare the slow base tokenizer. The exact repository's reviewed adapter
selects the fast class consistently in preflight and loading. Both outer and inner
index declarations must match that selection; other repositories or tokenizer
classes gain no alias permission.
Applying a loading alias preserves the complete three-field Modular component
declaration, including its pretrained origin and subfolder. A flat two-field
alias belongs only to a standard pipeline index; using it in a Modular index
would silently omit that component from upstream loading and registration.

The [joint AV artifact closures](../data/joint-av-artifact-closures.v1.json)
bind each workflow to one transformer partition, shared components, finite
metadata files and required notices. Pika includes its task adapter; MiniMax
does not. Their lookup is pure data and performs no model import, download or
installation. The newer file closure supplements the historical H3 review
without rewriting its earlier qualification state.

The ordinary model manager publishes these finite inventories through the existing
task artifact selections. Generic workflow models show a task selector; the
selected task alone determines Install/Repair and installed status. Acquisition
still uses the existing Hub planner, reservation, downloader, validation and
license acknowledgement flow. H3/Pika acquisition refuses missing, partial,
wildcard, mixed-partition or wrong-revision selections before planning or queuing
work. A selected file closure does not qualify a runtime, graph, Auto recipe or
public template. Text and first/last-frame tasks intentionally share their exact
transformer/adapter bytes; reference tasks use a separate partition.

No-weight tests cover selected starters, strict ownership, actual node dispatch,
inference mode, scheduler restoration, normalized decode, cache reuse and saved
graph connections. They do not prove video quality or memory feasibility. Native
acceptance must record the deployed source pair, immutable model, runtime, actual
inputs, task result, decoded dimensions/frame count, playback, repeat behavior,
cancellation and scoped release on the tested host. Keep logs and generated media
outside Git and retain unsuccessful attempts for diagnosis.

Video/audio export preserves every generated video frame. It trims or pads the
soundtrack to the video's duration instead of letting audio sample rounding end
the mux early. Retained-video audio muxing keeps its explicit duration choice;
temporary audio belongs in a newly created private directory, never a predictable
filename beside a borrowed source. Neither export path may remove user inputs.
