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

No-weight tests cover selected starters, strict ownership, actual node dispatch,
inference mode, scheduler restoration, normalized decode, cache reuse and saved
graph connections. They do not prove video quality or memory feasibility. Native
acceptance must record the deployed source pair, immutable model, runtime, actual
inputs, task result, decoded dimensions/frame count, playback, repeat behavior,
cancellation and scoped release on the tested host. Keep logs and generated media
outside Git and retain unsuccessful attempts for diagnosis.
