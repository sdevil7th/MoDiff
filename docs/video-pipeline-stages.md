# Video pipeline stages

Fresh selected video graphs use ordinary
**Model Setup → Encode Inputs → Generate → Video Output** stages. Generate groups
the actual sampler and decoder when their ownership and connections allow it.
Source preparation and optional upscaling remain independently visible. The stage
cards expose the settings and connections of their underlying executor nodes.
Use **Inspect implementation** to see those nodes, or **Edit stages** to separate
them on the canvas. Optional upscaling, separate soundtrack generation,
audio fitting and delivery remain visible operations when a template needs them.
Four stages is a starting shape, not a limit.

HunyuanVideo 1.5 Generate groups its actual sampling and decode nodes only when
their declared task, pipeline, state edge and component ownership agree. Image
Input remains separate. The decoder's optional VAE Tiling control temporarily
selects the official VAE's tiled decode and restores its preceding setting on
success or error. New 720p starters enable it; saved graphs with no selection
inherit their existing VAE setting. Offloading moves model weights and does not
remove the full-frame activation demand of an untiled decoder.

The separately declared `LTX25FullPipeline` facade uses the pinned LTX-2.5
repository's `transformer_full` with the official `LTX2ConditionPipeline` and its
Full scheduler recipe. The earlier distilled route is unchanged. The ordinary
Load/Generate Video interfaces expose text, opening-image, ordered-keyframe and
source-video conditioning with native audio output. Full controls include video
and audio guidance, STG, modality guidance, rescaling, selected guidance layers,
cross-timestep guidance and local prompt expansion. The shared execution recipe
owns VAE tiling; full residency and model CPU offload are the reviewed placement
choices. The facade does not declare source-audio conditioning, second-pass
upscale/LoRA, a diffusion decoder or latent output. Declared contracts and
no-weight tests do not establish live model or hardware qualification.

Ordinary setup includes pinned PyAV for the official image-conditioning H.264
recompression step. Preflight and runtime health check that PyAV can resolve the
`libx264` encoder and H.264 decoder. Repair this dependency with the ordinary
setup for your current accelerator. The adapter preserves the upstream
conditioning-compression defaults.

LTX long image-to-video starts with the distilled model's eight steps per
window, guidance of one, and a 128-token sequence length. The standard and
low-memory starters use the same sampling controls. A saved older draft
that reports an incompatible sampler can set Steps to 8 and Guidance to 1
in Generate Video.

AnimateDiff source, control and combined source/control starters use **Auto**
attention. Their loaders may reject a forced math backend. The current
recipes change only that explicit attention setting. Previously saved
execution specs retain their original identity and do not silently override
the user's selection. Source Video and Control Video use the same generic
video input component and present separately labelled file fields.

The AnimateDiff SD1.5 routes select the standard MotionAdapter safetensors file
from the pinned revision and load it at the requested FP16 precision. The
separately published FP16 filename contains different motion tensors and
produced dark frames in a controlled comparison. An installation containing
only the older file needs the standard file from Model Manager. Older run
records retain their actual artifact and source identity. AnimateLCM retains
its own FP16 adapter and spatial LoRA. The loader resolves both files through
the same immutable snapshot lookup as Model Manager, including secondary
cache locations.

AnimateDiff adapter readiness checks the catalog's standard weight file and
configuration in the snapshot used by the loader. A download receipt for an
older file variant alone does not mark the current workflow ready.

The built-in Latte text-to-video and CogVideoX video-to-video recipes
start with Auto attention, which uses the processors supported by their
installed pipelines. If a saved workflow reports that Math attention is
unsupported, select Auto in Model Setup.

Registered native-stage recipe descriptors retain their original versioned
selection metadata when only their visual grouping changes. Selection notes
participate in the strict template lock; editing explanatory wording must not
retrospectively relabel an unchanged executed recipe as a different one. The
existing LTX text, opening-image and ordered-reference declarations retain those
identities. This does not equate historical whole-pipeline or sequence calls with
new native-stage execution, relax proof locks, or qualify an unexecuted route.

Fresh multi-shot templates author each shot as a separate editable group. Inside
each shot, Image Input (when used), Encode Inputs, Generate and Save Shot are
ordinary cards. Winter Delivery and
The Last Signal upscale each shot before assembly; Before Sunrise and the
coffee sequence upscale after assembly. One shared Model Setup owns the
primary pipeline. Storyboard, ShotPrompt and ShotInputs are deterministic data
nodes, not model runners or another executor. Storyboard owns ordered text and
seed inheritance. Each real Generate stage owns its editable frame count and
conditioning; authored durations initialize those controls instead of silently
overriding later edits. One shared per-shot FPS value supplies generation and
Save Shot. Internal data-helper outputs without consumers are not public stage
sockets. Assemble or Assemble & Export exposes the ordered shot inputs and
timing or delivery controls in a single card. Ordering and explicit frame trim
remain internal operations.
Run Shot includes that shot's model and storyboard dependencies and excludes
other shots and final assembly. Saving a shot is a recovery side branch for
in-memory assembly, avoiding an extra lossy encode/decode in its render inputs.
The Gallery recorder accepts the resulting explicit-stage graphs and checks
every authored shot, its model/encoding/decode ownership, soundtrack, upscale
and final delivery. A partial Shot run cannot qualify a complete sequence.
Valid deterministic encoding/decoding/upscale cache receipts remain usable;
each sampler, saved shot and final assembly/export must execute freshly for
an independent run. Historical whole-operation receipt readers remain intact.

Video encoding and generation share guidance because CFG determines the
embedding branches. Each repeated shot has its own bounded advisory membership
for these paired controls, even when the pipeline is shared. Cross-model changes
retain the existing semantic transfer rules; pairing must not copy guidance
between unrelated model implementations. Common component/storyboard inputs
use the existing mirrored Block input contract rather than duplicate sockets.

Prepare Mask on fresh video-outpainting templates exposes two explicit choices.
Prepared keeps the source canvas unchanged and aligns the selected boundary
mask. Expand pads normalized RGB frames using the four borders and creates
white generated borders with a black protected source region. Only the active
mask input is read. Output geometry is wired to the real generation controls;
memory admission and cancellation happen before/through canvas allocation.
Saved Load Mask/Align Mask graphs keep their previous behavior.

Moonflower's versioned nine-scene recipe uses ten ordered opening references
(two for the garden-path shot), a 45-second song and an explicit 720-frame trim
at 16 FPS. The accepted film has no burned-in lyrics. The ordinary Lyrics stage
therefore defaults off for this recipe; enabling it requires an authored LRC
timeline. Other existing lyric nodes retain their enabled default. The shorter
five-scene recipe remains available as source history, and saved graphs are not
rewritten. Reconstructing authoring does not replace the accepted output or prove
that a newly rendered version matches it.

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

| Execution profile                | Tasks exposed by separate stages                            |
| -------------------------------- | ----------------------------------------------------------- |
| `ltx-video:direct`               | Text, opening image, ordered reference images, source video |
| `wan-text-to-video:direct`       | Text-to-video |
| `wan-video-to-video:direct`      | Source-video generation and color editing |
| `wan-vace:direct`                | The selected profile's native VACE conditioning tasks |
| `wan-22-image-to-video:direct`   | A14B opening image, with optional ending image |
| `wan-22-ti2v-5b:direct`          | Text-to-video                                               |
| `wan-22-ti2v-5b:image-direct:v1` | Opening image to video                                      |

The retained Wan 5B text route uses the upstream `WanPipeline`. The explicit
image profile uses the official `WanImageToVideoPipeline` with the same pinned
dense 5B checkpoint and its expanded-timestep configuration. It takes one opening
image through the official VAE; this checkpoint does not need CLIP components or
the larger model's second-transformer guidance. Text and image loaders have
separate backend identities, so a cached text owner cannot serve the image stages.
The new image selection grants no Auto, native-output or hardware qualification.
The existing text specification and saved graphs retain their identity.
Compiled older Blocks also retain their original loader-selector choices and
definition hashes. The raw loader and new image recipe expose the new backend
choice without adding it to those historical compiled schemas.
The reviewed [official I2V implementation](https://github.com/huggingface/diffusers/blob/7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941/src/diffusers/pipelines/wan/pipeline_wan_i2v.py)
declares the optional CLIP components and the dense model's expanded-timestep
conditioning; MoDiff delegates both source-image encoding and sampling to it.
Other video routes keep their
existing whole or native Modular operations. Saved whole-pipeline graphs are not
automatically rewritten. Stage availability is an authoring declaration; it does
not imply hardware qualification, Auto support or successful model execution.

Wan source-video generation delegates source VAE encoding and strength-adjusted
sampling to the official pipeline. That call derives frame count from the source
latents rather than accepting a duplicate `num_frames` argument. The staged
decoder retains that actual temporal geometry. A14B ownership additionally binds
both transformer experts and any optional image encoder/processor; replacing a
component invalidates earlier encodings. Its high- and low-noise guidance controls
remain distinct in the Generate stage and consumed-input evidence.

Wan first-and-last-frame workflows fit the closing image to the first image's
resolved geometry in both the vision and VAE encoders. The pinned Diffusers
crop passed dimensions in reverse order, which padded landscape inputs and
distorted the ending. The compatibility step center-crops to the requested
aspect ratio and resizes to the resolved dimensions. The existing source
limits still bound the fit, and the single-image branch keeps its own native
preparation. The First and Last Frame inputs and the saved graph format stay
the same.

VACE has a different genuine material boundary. Encode Inputs retains its frozen
video, aligned masks and optional references and runs official text encoding.
The existing generic Generate action receives that owned encoding. Its native
overlapping windows each decode internally so the previous generated last frame
can anchor the next masked window. There is no fabricated whole-clip latent
tensor or separate decoder. Seeds, neutralized generated regions, overlap removal
and the full delivered frame count retain the existing adapter's behavior. The
ordinary integrated Generate action remains available to saved graphs without
an encoded input. FPS belongs to delivery; VACE has no native FPS call argument.
Its whole-operation contract keeps `diffusion.generate_video`; the separately
selected encoded-stage contract is `diffusion.render_video`. Both use the same
generic Generate executor. An unbound model draft retains the whole operation,
while a curated stage selection gets Encode Inputs and the encoded consumer.
Partial mixtures are rejected instead of replacing either contract implicitly.

The encoded render contract requires `encoded_inputs`; its raw source, mask and
reference sockets are inactive there. Those media remain required by the selected
Encode Inputs task and by historical whole-operation calls. Readiness uses the
current backend bundle declaration for an exact literal operation selection,
including restored workflows, without rewriting saved schemas. A missing,
disabled or incompatible encoder still blocks Run. Saved advisory port metadata
cannot relax the current backend declaration.

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

`VideoConditioning.OutpaintCanvas` is deterministic preparation for an expanded
video canvas. It takes uniform prepared RGB frames, places them unchanged at
explicit left/top offsets, and creates same-sized white-generate/black-source
boundary masks. Left/right/top/bottom controls add borders rather than stretching
the source. Frame order and count remain unchanged; generation dimensions come
from its canvas outputs. Model-specific spatial multiples still belong to the
selected generation contract. Prepared source/mask inputs remain a separate
compatible route and must not be padded again implicitly.

Preparation checks observed available host RAM before allocating both output
collections, retaining the existing CPU reserve of at least 4 GiB or ten percent
of physical RAM. This allocation guard does not impose a guessed model-size
limit or grant Auto/hardware qualification. Non-RGB or mixed-geometry input asks
for normal video normalization first. The source frames are not modified, masks
are independent per-frame objects, and cancellation uses the existing node path.

The ordered-reference assembler requires only the medium selected by
`reference_kind`. The initial ordered list is optional; video soundtrack is
optional in Video mode. Image, Video and Audio modes respectively require an
image, decoded video frames or decoded audio. Published fields use
`required: false` with `requiredWhen: {"field": "reference_kind", "equals": ...}`.
Client Fix and preflight evaluate this read-only condition against a literal
selector using the current registry, including for older saved nodes. A wired
selector is not evaluated during inspection; execution still validates its
actual selected value and media. No saved schema, values or wires are rewritten.
The final reference consumer also retains the no-audio-only constraint.

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

The existing whole-pipeline Wan 2.1 1.3B sequence can also receive static Auto
storage admission through `GenerateSequence → Compose →` the pinned tiled x2
upscaler. It requires one accepted immutable model owner, its sole sequence
consumer, two to six literal prompt/seed shots, native PIL output, one video per
shot and one collection entering Compose. Geometry and frames per shot remain
the accepted owner's literal controls; additional clips, conditioned inputs,
external latents, mutable model pins and dynamic shape controls are refused.
Composition's bounded crossfade cannot increase the sum of source frames.

This route budgets every shot, both retained RGB source/composition collections,
float32 input/output tensors, converted x2 frames and one complete tiled CPU
assembly. This conservative known-storage allowance is not an activation or
loading peak. It does not grant model fit or hardware qualification: the existing
capacity floors, generating-owner release schedule and actual free-memory
resample before upscaler allocation remain required. The native LTX stage route
retains its separate unchanged storage contract.
When the generating model has only a machine-capacity tier rather than a complete
working-memory demand, Auto releases it before loading the independent image or
video upscaler, even if the upscaler's known buffers fit available memory. An
overlapping graph that cannot release that owner remains held. Reviewed dependent
components such as ControlNet retain their existing placement rules.

Saved whole-pipeline Wan VACE text-to-video graphs can receive the same static
upscaler storage forecast when the exact reviewed checkpoint owns a single
text-only PIL render. The forecast inspects literal, aligned dimensions and
frame count and refuses external conditioning, embeddings or latents. The
pipeline encodes its prompt internally. Saved attention settings still pass
through the separate Auto execution-recipe checks.

The client uses the backend's exact selected starter and
`generic-video-stages:v1` authoring recipe. The retained creative settings and
ancillary branches remain intact. A decoder and file-backed exporter are grouped
only when their actual connections allow it. An intervening upscaler or independent
audio branch remains distinct.

Fresh ordinary video stages and ACE audio recipes bind attention to the authored
form choice, defaulting to official automatic attention. They do not force math
attention merely because an older whole-pipeline specification selected it.
ACE selects the explicit `generic-audio-recipe:v1` authoring projection for its
four integrated tasks; the retained source specifications and compiled Block
identities remain unchanged. This projection adds no new execution stage.
Explicit math/flash choices are retained and checked through the existing backend
recipe and installed-library boundary. Saved graphs keep their literal policy;
there is no automatic migration, output-parity claim or additional hardware
qualification. A capacity tier still cannot admit forced math attention without
an explicit working-memory declaration.

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

First/last-frame preparation carries the selected frame count in official
`PipelineState`, whose downstream before-denoise block consumes it. The keyframe
preprocessor itself accepts images and geometry, so passing frame count to that
block would silently ignore the control. The reference preprocessor instead
consumes frame count directly for reference validation/alignment. Both paths
retain the existing explicit state connection and official duration alignment.

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

The Wan-family and both LTX Video loaders resolve Hub selections to the exact installed
snapshot through the shared managed-cache resolver. Model discovery and execution
therefore search the same configured and secondary cache roots. Wan loaders load their
float32 VAE and the remaining pipeline from that same immutable directory; the
original repository and revision remain the execution identity. Local selections
retain their explicit path. Loading neither copies weights between caches nor
downloads a replacement when the selected revision is absent.

Latte also loads from its exact managed snapshot. The pinned model index
selects the standard VAE. Passing a Hub repository ID to Diffusers can make
its repository-tree check demand the separate temporal decoder even after
Model Manager verified every selected file. The installed snapshot directory
keeps loading restricted to the components named by the pinned pipeline.

Workflow Auto inspects Storyboard, Shot Inputs and Shot Prompt through their
shared deterministic scalar logic, without constructing or executing a node.
Numeric widget strings use the ordinary declared scalar conversion. Prepared
outpaint geometry comes from the exact Normalize source; expansion adds the
declared borders. Runtime preparation still checks the actual supplied values.

The pinned x2 Spandrel storage forecast also follows explicit native LTX shot
collections and Wan/VACE video outputs. It requires the same accepted immutable
model owner, matching encoder/sampler/decoder bindings, PIL output and finite
geometry. Compositions sum every source frame conservatively, before crossfade
deductions. This is retained-frame storage accounting, not a measured activation
peak, arbitrary-graph permission or hardware qualification. Unknown sources,
borrowed bindings and unresolved controls remain refused. The original closed
single-shot LTX storage contract keeps its existing identity.

## Editing existing video

Use **Process Video** for frame extraction, trim, reversal, interpolation, joins
and tiling. Select files or connect a source clip or an ordered collection.
A connected source takes precedence over the saved file selection. Files use
their own frame rate. For decoded frames, set the frame rate on the card.
Decoded frames are written incrementally to a temporary lossless clip that is
removed after the operation. File sources stay on disk.

Trim and interpolation retain embedded sound. Reverse follows the reversed
picture. A join aligns each first audio track to its clip, inserts silence for
silent clips and crossfades sound over the same interval as the picture.
Tiling mixes available first tracks over the chosen output duration. Extracting
frames returns images, indices and times. The existing bounds on frame count,
output size, interpolation and reverse buffering apply to both input forms.
