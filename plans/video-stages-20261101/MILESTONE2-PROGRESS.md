# MiniMax/Pika progress and evidence boundaries

Updated 2 November 2026 IST. Work remains on the same two PRs, with source authored
in the canonical Windows pair and native models on Linux. The first LTX/Wan
testing app has not been replaced by this in-progress milestone.

## Completed in the current slice

| Change                  | Evidence and scope                                                                                                                                                                                                                      |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| H3 Model Setup          | Native task `RotZTltUc8G4`, 118.70 seconds; loading only                                                                                                                                                                                |
| Conditioner entrypoints | Source-declared leaf hooks for the actual nested Qwen3-VL call; tiny numerical red/green regression and four exact-provider Linux checks                                                                                                |
| Audio VAE placement     | Honor the official group-offload opt-out; reviewed resident placement for the small float32 H3/Pika audio VAE, with actual per-component cache policy. 193 focused checks, one skip, 32 subtests; four exact-provider Linux checks      |
| H3 engineering export   | Task `7BojyLbXZkAp`, backend `b770e165c7b665ba9006ec98df89e744841d8543`; 512×288, 124 frames, 24 fps, two sigma points / one model evaluation. Completed in 552.83 seconds without a resource guard                                     |
| Actual saved media      | SHA-256 `35468d91636462bbd701df90ca6479e94432de14972bc1da23d63e3c1e37c65c`; 2,687,607 bytes. All 124 frames decode and differ; stereo 32 kHz audio is finite and non-silent. Exact cache-owner/file bytes agree                         |
| Same-compute reuse      | Task `40RBQUbQ2JpZ`, 0.375 seconds; actual WebSocket events report all five nodes cached. One genuine persisted output record. This is API/executor evidence, not a visible browser or quality signoff                                  |
| Inline adapter message  | LTX/Wan Model Setup accurately reports its unsupported inline LoRA editor instead of blaming valid consumers. 387 focused checks and two mounted-control browser cases; graph/export unchanged                                          |
| Video input history     | Record LTX condition strength separately from denoise strength and identify the ordered frozen PIL source-video sequence/count. 44 backend checks; 43 client receipt/history checks. No path/media-byte capture or old-history backfill |
| Export preservation     | Borrowed audio and prior completed outputs survive failed muxing; owned temporary intermediates, alias rejection and atomic destination replacement. Previously retained tiny real-media regressions                                    |

The engineering clip is blurred and does **not** meet the recognizable-sailboat
brief. The first raw API graph also omitted the normal client preview
`display`/`sourceKey` metadata, so its original persisted outputs list stays empty.
The separate cached replay adds that metadata without changing compute inputs;
its genuine receipt is not retroactively assigned to the first task.

## In progress and still open

- First H3 quality recipe: 960×544, 124 frames at 24 fps, seed 20261102, 50 sigma
  points / 49 evaluations. Graph SHA-256
  `f6a7fb473ee0228f661f3b151459352885aa66c8b9f4678b347514468c389e95`.
  The same prompt/model/offload owner is retained. Estimated 1–3 hours from the
  lower-resolution measurements; larger-canvas timing is not yet measured.
- Pika: 35 selected files, 146,806,845,173 bytes, downloading at its immutable
  revision. No Pika pretrained output exists yet. Three task bindings, public
  source-accurate hierarchy/catalog linking and actual adapter/sampler acceptance
  remain open.
- H3 keyframe/reference routes, the remaining Wan image route, finishing stages,
  all twelve final creative recipes, final lifecycle and visible browser checks.
- Complete current backend/client gates, exact paired bundle, protected refresh,
  supported/held matrix and final manual handoff. The earlier complete backend
  attempt had four stale image-catalog fingerprint failures; the semantic-preserving
  refresh and affected tests passed. It is not a complete current gate PASS.
- Public clip publication requires exact-file review; existing image approvals
  do not approve these videos.

## Logging and mistakes retained

Raw attempts, source/dependency identities, selected-file receipts, resource
samples, terminal API responses, original WebSocket events, media hashes and
commands remain in the private campaign evidence. The append-only event log
records failures and harness corrections rather than replacing them with a
later PASS. No private logs, media, model weights or host configuration belong
in Git.

The fixed native defects were empty optional-adapter dispatch, conditioner calls
bypassing outer group hooks and an official audio-VAE offload opt-out being
ignored. Initial overly coarse transient-swap guards and missing test-preview
metadata were harness mistakes, not model fit or product quality failures.
The resource monitor keeps a 20 GiB available-RAM floor, bounded worker swap,
175 GB owned scratch and 160 GB free-filesystem reserve. Concurrent Pika download
growth is measured separately from the owned model scratch. Do not override
official model flags, modify model math or rerun unchanged failing allocations.
