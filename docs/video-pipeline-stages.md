# Video pipeline stages

Fresh selected LTX 0.9.8 distilled and Wan 2.2 5B text-to-video graphs use ordinary
**Model Setup → Encode Inputs → Denoise → Video Output** stages. These are real
registered nodes in the existing executor. Expanding a visual group exposes its
actual nodes and connections. Optional upscaling, separate soundtrack generation,
audio fitting and delivery remain visible operations when a template needs them.
Four stages is a starting shape, not a limit.

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

The separately reviewed MiniMax H3 schema keeps its video and audio VAEs in
float32 while the default component precision remains bfloat16. The selected
`transformer_ref` partition and `audio_vae` participate in the existing owned
group-offload path; direct audio-VAE encode/decode uses leaf hooks like the video
VAE. This is loading-policy coverage, not measured H3 resource fit or live
video/audio acceptance. Pika's source runtime remains a separate candidate and
does not inherit these model qualification claims.

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

The [joint AV artifact closures](../data/joint-av-artifact-closures.v1.json)
bind each workflow to one transformer partition, shared components, finite
metadata files and required notices. Pika includes its task adapter; MiniMax
does not. Their lookup is pure data and performs no model import, download or
installation. The newer file closure supplements the historical H3 review
without rewriting its earlier qualification state.

No-weight tests cover selected starters, strict ownership, actual node dispatch,
inference mode, scheduler restoration, normalized decode, cache reuse and saved
graph connections. They do not prove video quality or memory feasibility. Native
acceptance must record the deployed source pair, immutable model, runtime, actual
inputs, task result, decoded dimensions/frame count, playback, repeat behavior,
cancellation and scoped release on the tested host. Keep logs and generated media
outside Git and retain unsuccessful attempts for diagnosis.
