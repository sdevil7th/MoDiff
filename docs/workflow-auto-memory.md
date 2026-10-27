# Workflow Auto memory contracts

Workflow Auto plans the exported execution graph and runs it through the existing
backend executor. Shared loaders count once; distinct loader identities count as
independent model owners, even when they select the same repository. Planning
never loads models, installs packages, changes prompts or reduces creative
settings to make a workflow fit.

## Bounded public FLUX 3 recipe

The exact public Distilled and Base profiles declare a Windows x86_64 recipe
for primary dedicated NVIDIA CUDA: 512×512 output, batch one, explicit FP8r
storage with BF16 compute, CPU weight loading and model CPU offload. Distilled
retains 16 steps; Base requires 50 steps, ordinary classifier-free guidance 3.5
and rescale zero. The declared text window is at most 8192 tokens. Native,
NVFP4, additional quantization, other placements and larger workloads receive
no permission from this FP8r declaration. The separately selected NVFP4/text
NF4 recipe below has its own contract. Connected LoRAs retain their independent
resource and placement restrictions.

Architecture uses the existing runtime normalization, including Windows' AMD64
spelling for x86_64. Missing or unknown architecture remains unsupported rather
than borrowing the planning process's host architecture.

Reference editing requires one or two actual prepared PIL images, each at most
1024 pixels per side, with at most 512² + 1024² total pixels after the pinned
16-pixel center crop. Planning inspects the exact direct local Load Image source
inside configured work/data roots, with at most two 32 MiB files. It never
executes a supplier or downloads media. Unknown/computed/remote source geometry
refuses before submission. A bounded RGB/order/geometry witness is checked
again against the actual Image Encode argument before its resources allocate.
Changing or replacing a reference invalidates that witness. The encoder's
ordinary opaque binding still validates its actual processed conditioning.

Admission observes every file in the exact FP8r selection, including the variant
checkpoint. It requires the reviewed active provider, packed tensor/scales and
FP32 norm census, dense text/VAE census, conservative whole-file source/target
loading overlap and per-layer FP32 dequantization reserve. Existing actual free
RAM/VRAM, allocator headroom and owner-lifetime checks remain mandatory. This
can refuse a cold run that completed in Expert mode; logical storage and
conservative overlap are not measured physical commitment or peak inference
demand. A cached file, prior Expert output and a source declaration do not
establish native qualification. Actual Automatic output and visual acceptance
remain separate proof obligations.

### Explicit NVFP4 and parent text-encoder NF4

The public Distilled and Base profiles also name
`flux3-nvfp4-text-nf4-windows-model-cpu:v1`. This is an explicit ordinary graph
variant, never an automatic precision substitution or a changed starter default.
It keeps the same immutable parent repository/revision and bounded T2I or
one/two-reference edit workload above, with BF16 compute, `cuda:0`, model CPU
offload, batch one, 512×512 output and no LoRAs or component replacements.

Add the existing **Quantization Config** node and select the exact parent Hub
repository, **Subfolder** and **Component** `text_encoder`, **Quant Type**
`bnb_4bit`, **4-bit Quant Type** `nf4`, **Compute Dtype** `bfloat16`, and
**Double Quant** enabled. Set **Skip Module Names** to `model.visual,lm_head`
and leave the legacy **Skip Modules** list empty. Connect its
`quantization_config` output to the parent **Quant Config** input and explicitly
choose transformer **Weight Format** `nvfp4`. Save this as a separate authored
variant before normal Automatic Run. The parent prompt, layout, reference order,
seed and step count are unchanged. The Studio Expert quantization menu is not
the selection path for this Auto recipe.

Planning inspects only that exact built-in producer and literal controls; it
does not run a configuration supplier, load layers, instantiate a model or
install packages. The component map is separately identified from the
transformer's scalar quantization mode. Source graph, producer controls, exact
resolved Transformers NF4 config, parent identity and loading policy are bound
again at dispatch and in cache/history identity. Arbitrary producers, changed
exclusions, extra component mappings and unreviewed connected controls refuse.
After successful loading, the existing consumed `quantConfig` receipt records a
bounded string list describing both actual component configurations, loading
devices, validated packed encoder state, recipe and source signature. This
snapshot preserves the connected producer origin through output history; it
does not serialize executable configuration objects.
The text encoder's language Linear layers use NF4; its vision module and output
head remain dense. The official BFL transformer configuration remains separate.

The reviewed component loader explicitly stages the NF4 text encoder on primary
CUDA before model CPU offload; the BFL transformer retains CPU cold loading.
For a fresh complete owner, the validated packed text encoder stays on CUDA
while the official transformer and VAE loaders finish on CPU. Only then does
the existing manager register the text encoder and install its CPU offload
hooks. This removes the packed text owner's CPU storage from the transformer's
whole-file source/destination conversion overlap. No model forward or cache
publication occurs before final registration and CPU packed-state validation.
The internal loading schedule is bound to the descriptor, history/cache
identity and phase record; it does not change any authored workflow control.
Partial/reused component sets and preexisting neural owners cannot borrow the
smaller fresh-loading phase budget.
The parent `weight_loading_mode=cpu` therefore does not claim that all component
conversion occurs on CPU. Admission counts the exact dense and packed component
storage, quantization state, source/conversion overlap, CUDA staging and
per-layer dequantization reserves under the existing RAM/VRAM headroom and owner
lifetime rules. Missing evidence or insufficient actual free memory refuses
without replacing the authored format or reducing the workload. The existing
reviewed provider and activated bitsandbytes runtime are mandatory.

The host phase census retains the complete text source, complete BFL source
and destination, VAE conversion overlap and final combined CPU parameters.
It takes their maximum under this enforced loading order, then adds the same
max(4 GiB,10% system RAM) floor. GPU staging and its max(2 GiB,10%) floor remain
unchanged. Fresh actual capacity is checked before every allocation and before
final text CPU registration; no mmap sharing or projected allocator reclamation
counts as free memory. A loading failure or cancellation destroys only this
attempt's fresh, exclusively owned staged components and clears retaining
exception frames. It does not register a fallback owner or publish a successful
cache. These allocation-lifetime checks are not measured inference qualification.

This declaration does not qualify legacy/private profiles, other precisions,
hardware targets, native output, Gallery or templates. A successful cold normal
Automatic run and visual acceptance remain required independent evidence.

## Machine capacity and working demand

A recipe with `memorySemantics: machine_capacity` describes the machine class
required for that recipe. Its RAM and accelerator tiers are checked against
machine capacity. The largest applicable tier is retained across owners; adding
those tiers together would incorrectly treat machine capacity as model demand.

An explicit `workingMemoryRequirements` declaration instead bounds additional
working demand for the selected workload and offload mode. Independent owners'
demands and connected adapter storage are additive. Compatible, live cached
weight storage can receive credit once, within its owner's explicit budget.
Free host memory and accelerator memory remain separate on dedicated GPUs and
share one pool on shared-memory accelerators.

Each cached owner's eligible storage is retained until the planner selects
compatible current owners. Physical storage is then credited once across that
selection. An unrelated earlier cache record cannot consume a current owner's
credit, and per-owner storage totals must not be summed to estimate reclamation.

A fully resident `none` recipe uses the disk requirement of its accepted
residency tier, rather than inheriting the minimum tier's SSD-offload floor.
Actual disk-offload placement and missing-model download requirements retain
their own capacity checks. This removes a false disk blocker without promising
that unreviewed models fit in memory.

A measured free-memory value of zero remains zero. CUDA, XPU and MPS snapshots
fall back to another reading only when the preferred reading is unavailable;
truth-value checks must not replace a fully occupied device's zero with older
free-memory telemetry. The next-owner capacity check consumes that preserved
reading and blocks allocation when the requested headroom is unavailable.

Some reviewed recipes have a machine tier but no complete working-memory demand.
They keep the existing single-owner runtime headroom policy: at least 4 GiB or
10% of total system RAM, and, when applicable, 2 GiB or 10% of accelerator
capacity. These floors are safety headroom, not measured model weights or peak
inference demand. A single floor cannot establish that several independently
loaded models fit together.

## Known parameter storage and loading phases

For an already validated immutable native image snapshot, Auto reads bounded
Safetensors headers and shard indexes to count parameter storage at the requested
unquantized dtype. Candidate `parameterStorageLowerBound` records contain the
exact revision, dtype, per-component byte counts, total and largest component,
with `proofScope: parameter_storage_lower_bound_only`. Reading these headers
does not construct a model or grant artifact, runtime or memory qualification.
Unsupported quantization, missing headers and unknown layouts retain the existing
unknown-demand policy. Audio and video host-memory admission is unchanged.
Image scope follows repository-aware backend capability output kinds and their
task overrides. A legacy profile without that capability definition must match
its exact declared repository and execution specification with an image Preview
terminal; pipeline class names do not establish modality.

The ordinary native loader has these proven CPU storage minima:

| Placement | CPU parameter minimum | `parameterResidencyPhase` |
| --- | --- | --- |
| CPU execution, model CPU offload (including `auto_cpu`), group CPU offload | Sum of selected component weights | `cpu_resident` |
| Accelerator execution with `none` offload | Sum of component weights loaded before accelerator placement | `cpu_loading` |
| Incremental group disk offload | Largest entire component loaded before its disk hook is installed | `cpu_loading` |

Requested-dtype storage alone does not cover cold loading. The companion
`parameterConversionReserve` binds source payloads, converted targets and shard
layout to the same revision, dtype and component census, with
`proofScope: source_requested_loading_overlap_only`. Native image admission
requires both records. A missing or invalid companion cannot fall back to the
smaller requested-dtype budget. These records describe conservative storage
overlap, rather than measured anonymous commitment or a complete inference peak.

For serial Diffusers loading, each shard's source and all of its converted
tensors coexist with the requested tensors retained from preceding shards.
Parallel or unknown loader contracts reserve all source and converted payloads.
Transformers loading keeps that conservative reserve. The effective Diffusers
flag comes from its captured ModelMixin constant after import, rather than a
later environment edit. Retained CPU components add their requested storage
beside the largest component's conversion excess. Disk LoRA targets that remain
staged for injection need the same accounting across every declared target;
adapter storage remains separately additive.

The optional advanced native loader control `weight_loading_mode` defaults to
`cpu`. A supported `cuda` cold-loading candidate uses the official component
loader and installs the existing disk hooks before loading the next component.
Its CPU reserve still includes shard conversion and disk serialization copies.
Its separate cold GPU reserve holds the entire largest component in addition
to the unchanged runtime headroom. The smaller inference headroom alone cannot
admit CUDA loading. Initial eligibility is limited to explicitly reviewed dense
BF16 component classes on primary dedicated NVIDIA CUDA, immutable native image
snapshots and group disk offload, without LoRAs, quantization or external
components. The actual loader rechecks class, configuration and free memory
before allocation. An unsupported choice is refused without a loading fallback.

Auto applies this control through the ordinary graph patch and shared-control
contracts. Legacy graphs without the field retain CPU loading; Auto never
inserts the missing field. Cold conversion and GPU staging can be waived only
for a complete compatible successful loader cache hit. Partial component
ownership, disk placeholders and logical tensor sizes cannot establish that
exception. A changed loading mode changes ownership identity; release schedules
and dispatch-time free-memory checks retain the full cold requirements.

Native image disk LoRAs require private reviewed component targets matching the
installed pipeline's official `_lora_loadable_modules`. The loader offloads other
components first, loads all adapter targets last, calls the official LoRA loading,
activation and scheduler APIs, then installs fresh disk hooks on those targets.
Its cold parameter minimum is the greater of the sum of all staged targets and
the largest remaining component. Auto refuses disk LoRAs with missing target
declarations, incomplete target headers or unsupported quantized storage. This
minimum still excludes adapter storage, hook placeholders and loading overhead.
The separate disk parameter allowance counts base component storage only. Official
disk hooks also serialize injected adapter tensors and Safetensors metadata;
their exact serialized demand is not established by the adapter RAM/VRAM budget.
Neither base disk allowance nor the staging bound proves complete disk or RAM fit.

An exact successful disk owner may remain cached for unchanged graph and prompt
edits. Add, remove, order, scale, name, artifact or scheduler changes, other loader
input changes, explicit invalidation, implementation changes and partial component
eviction destroy the affected owner and its dependent model references before
replacement allocation. Successful reuse is bound weakly to the actual loader,
output, implementation, inputs and exclusive registered target objects; failed
or partial adapter loading cannot publish it. Shared mutation targets fail closed.
No custom supplier runs during inspection. Connected controls resolve through
ordinary graph dispatch; unknown preparation values require cold teardown.
These private target and cache identities grant neither qualification nor a new
audio/video feature, and do not change the downstream-adapter Auto boundary.

Every fresh disk-hook load directory has a process-local lease shared by its
component, child modules and official hook/group/config objects. Scoped owner
release collects their references and removes only that exact unchanged
directory before replacement allocation. Retained shared components, child
callbacks, copied configurations and adopted owners keep their files. Opaque
graph IDs, including Block V2 projections, map to bounded local storage names;
scope/component segments and directory ancestors reject path indirection.
Removal verifies the absolute cache boundary and filesystem identities.
Cleanup failures preserve files and are logged; disk admission still observes
the actual remaining free space. No reclaimable-byte credit is invented.

Explicit idle/global cleanup collects model and hook cycles before removing
unleased remnants, and refuses file deletion while any live lease remains.
Ordinary drained idle shutdown releases retained graph caches through the same
model executor and cache lease before closing that executor. Busy or undrained
work is preserved; external live hook users cause a logged cleanup error.
Shutdown and explicit idle cleanup own the existing teardown guard before
draining earlier resource probes. Delayed shielded resources, status and system
statistics builders use cached accelerator metadata during destruction. Status
also requires the exact previously validated runtime profile and observes only
already loaded packages; unavailable or stale metadata cannot trigger a fallback
device probe or imply readiness. Missing prior required package observations
return the existing `409 runtime_cleanup_busy` wait response rather than claim
that installed dependencies need repair. Valid status labels its hardware as
`cached_during_cleanup`. Cleanup-owned before/after CUDA diagnostics stay
fresh under the shared resource lock; outside readers retain cached observations.
Interpreter-exit finalization is disabled and startup does not sweep files, so
forced termination, undrained shutdown or cleanup failure can leave remnants.
The existing explicit idle cleanup can remove them once no live hook ownership
remains. Its file count excludes files already removed by lease finalization.
Tiny CPU hook/lifecycle tests do not establish live CUDA output, complete
serialized demand or hardware fit.

Owners publish this minimum in `parameterResidencyRequirements`. Cold RAM demand
is at least the greater of the declared working demand and this minimum.
For unknown working demand, the existing RAM headroom is additional to known
parameter storage; it is not absorbed into the weight count. Connected adapter
storage remains additive. The lifetime schedule uses these cold bounds and
checks actual free memory after owner release before loading the next owner.

On a dedicated NVIDIA CUDA primary device (`cuda` or `cuda:0`), known native
image parameter storage also bounds cold VRAM demand: `none` needs the sum of
component weights, and `model_cpu` (including `auto_cpu`) needs at least the
largest complete component. Unknown working demand still needs the existing
accelerator headroom in addition to those weights. Explicit working budgets
remain the greater of their declared demand and the parameter minimum. An exact
complete live owner can credit only its own measured unique GPU weight storage,
capped by that owner's demand; CPU bytes cannot offset VRAM. Observed free bytes,
including zero, remain the capacity authority. Allocator allocated/reserved
counters grant no free-memory credit. Group GPU peaks remain unknown. Other
CUDA indices, ROCm, shared-memory accelerators, audio/video and unsupported
quantization retain their existing GPU admission policy.

`cpu_resident` owners receive only the existing measured, unique live CPU storage
credit, capped by their RAM demand. Accelerator storage cannot offset their CPU
minimum. A `cpu_loading` owner also publishes `parameterLoadingAdditionalBytes`,
the portion of its cold loading minimum above its declared working RAM demand.
That portion is waived only for an exact complete successful loader cache hit:
the output, upstream inputs, implementation, native pipeline, registered required
component objects, manager collection and supporting model ownership must remain
live and unchanged. These process-local weak identity records come from ordinary
successful dispatch, never client hardware or receipt fields. A missing or
replaced component, incomplete output or released owner cannot grant the waiver.
Declared warm working demand and runtime headroom remain required.

Group disk hooks use CPU placeholder tensors after writing weights to disk.
Their logical tensor sizes do not prove retained physical weight residency;
Auto therefore does not use those sizes as warm CPU storage credit or as a
forecast of physical RAM reclaimed by clearing caches. The snapshot excludes
CPU storage belonging to the installed official accelerator-backed disk groups,
including shared aliases of that storage; unrelated real CPU weights and live
GPU weights retain their existing unique-storage accounting. A stale owner
record cannot restore placeholder credit. The disk loading minimum does not
include all residual components, placeholder commit,
conversion buffers, loading overhead or inference activations. Neither this
minimum nor the other parameter bounds establish complete peak demand or a new
reviewed recipe. Native sequential CPU offload remains unsupported.

For example, 19.11 GiB of known cold CPU weights with 4 GiB of runtime headroom
cannot be admitted on 13 GiB of free RAM merely because the machine satisfies
an 8 GiB capacity tier. An unchanged complete warm owner can receive the credit
described above, while a changed policy or cleared cache pays the cold demand
again. Dispatch replans before model execution and after cache preparation.

## Native image placement selection

For declared CUDA native image profiles, Auto exposes the existing supported
`none`, `model_cpu`, `group_cpu` and `group_disk` recipes as distinct candidates.
The original preferred candidate keeps its ID; an alternative has a stable
placement suffix and `placementAlternativeOf` identifying that original. No
repository, revision, dtype, quantization or creative setting is substituted.
An exact qualified offload override remains pinned to its reviewed mode.

An already admitted default that needs no cache preparation keeps its
placement. A proposal that depends on projected cache destruction must not
preempt an eligible exact complete warm native image owner that fits actual
available memory without cleanup or a release schedule. Auto checks that warm
alternative through the same full admission function before choosing a cold
cleanup proposal. Incomplete, changed, evicted or scheduled ownership grants no
warm preference. When known cold RAM or VRAM demand blocks the preferred plan,
Auto evaluates the declared alternatives
through the same whole-graph admission function, including independent owners,
adapter budgets, connected controls, cache preparation and the release schedule.
The search performs at most 64 complete evaluations and shares one trusted
hardware snapshot plus immutable recipe and header reads. It uses private
structured resource-failure categories;
request fields cannot force a candidate or supply warm-owner authority. A search
that cannot admit the whole graph returns the original refusal and no patches.
Unknown headers keep the previous unknown-demand policy and preference.
Progressive search compares actual whole-graph RAM and VRAM peaks first, then
private sums of cold CPU and GPU owner demand. The sums allow several sequential
owners to improve across an unchanged peak; they do not replace peak admission
or grant credit. Intermediate changes may retain only RAM/VRAM resource issues,
never disk, artifact, source, connected-control or other validation failures.

A known dense group-disk owner also declares `parameterDiskRequirements`, the
parameter-storage allowance for its component set. Cold disk demand is at least
the greater of that allowance and its existing declared disk requirement.
`parameterDiskAdditionalBytes` is only the extra allowance above that requirement.
An exact complete warm owner can waive that extra cold serialization demand,
because its existing hooks reuse the files; it still needs the original declared
disk free space. A scheduled or released owner pays the cold allowance again.
LoRA conversion and dense-update RAM budgets remain additive and do not become
claims about serialized adapter bytes. Disk loading overhead, residual CPU
components, disk placeholders and activation peaks remain unmeasured; these
minimum checks do not establish hardware fit or promote qualification. Audio,
video, quantized routes and unsupported native sequential offload retain their
existing selection behavior.

## Observed CUDA allocator recovery and busy inspection

For relevant raw VRAM pressure, the existing owned model thread may release
unused CUDA allocator blocks once and take a fresh actual hardware sample before
replanning the same graph. This requires the existing model/cache lease, model
I/O ownership, current callback-thread binding, process source identity and a
private attempt budget. The budget is shared by dispatch-time data/cache/policy
replans and is consumed even if the clear fails. No model owner is destroyed by
this step; there is no global unload, IPC collection, alternate executor or
client-requested allocator authority.

A known cold primary CUDA parameter refusal can request this one sample. An
exact complete warm unquantized native image owner with validated dense headers
can also request it for raw GPU headroom pressure, including `group_cpu` and
`group_disk`. This preserves a compatible warm placement when unused reserved
blocks alone caused pressure. Group GPU parameter peaks remain unknown, and
cold/incomplete/stale group owners cannot request a clear. The original raw
resource-only failure is captured before projected cache preparation or an
admitted cold alternative; unrelated admitted cold plans do not request a clear.
Final admission uses actual new free bytes and all normal checks, not reserved
counters or a promise that the clear will help.

Idle HTTP planning uses those existing leases and the same model thread. If work
becomes busy before cache/model-I/O ownership, it returns cached advice without
waiting for active work or invoking a device probe. Cancellation retains the
lease until its threaded callback completes. Busy advice requires a previously
successful full `runtime_profile` validation bound to this instance, executable,
environment, installed dependency identity and the small reviewed profile and
receipt metadata inputs. Missing, stale, malformed or unreadable validation
identity produces an actionable request to wait and plan again. Cached hardware
alone is not readiness proof. Busy inspection grants no live owner credit and
cannot clear CUDA; dispatch and idle validation still perform their actual
existing checks. These contract tests are not live allocation or model-fit
qualification.

## Independent model owners

When one or more primary owners lack a complete working demand, a workflow with
several primary owners can use Auto only when their lifetimes can be separated
through the implemented release schedule. This applies even on a large GPU or
when all loaders already have compatible cached models.

The existing lifetime planner can finish an owner's ready consumers before
loading another independent owner. The executor starts a scheduled attempt from
an empty model cache, detaches downstream material outputs, destroys expired
model ownership and checks actual free memory before the next owner loads.
If a retained output still holds opaque or device-backed model state, release
fails visibly before the next allocation. Projected reclamation never becomes
actual available memory.

When an unmeasured owner must overlap another primary owner, Auto reports that
combined model fit is unresolved. The user can separate the work into sequential
image stages, use a reviewed complete working-memory recipe, or select Custom
memory for the authored workflow. Custom Python does not establish that early
owner eviction is safe. Opaque loop execution also retains its owners until the
loop finishes; the planner does not insert destructive releases into its body.

Examples:

| Workflow                                                                         | Auto behavior                                                                                                 |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| One loader supplies two image-generation branches                                | Count the shared owner once and retain compatible model reuse.                                                |
| Two independent image-generation branches, with unmeasured primary demands       | Finish and release the first owner before loading the second, preserving their outputs.                       |
| One generated image feeds another model's editing stage                          | Preserve the detached image and use the release/recheck schedule when the primary demands are unmeasured.     |
| One consumer requires two primary models at once and either demand is unmeasured | Report unresolved combined fit; do not infer fit from machine tiers or free headroom.                         |
| Several owners have complete reviewed working demands                            | Retain them when their combined demand fits; otherwise use the existing safe lifetime schedule when possible. |

Reviewed auxiliary weights and LoRA storage keep their existing explicit
additive contracts. An auxiliary component attached to one model does not
become a second independent primary model solely because it has its own loader.
This distinction does not waive auxiliary identity, storage or dispatch checks.

Native model CPU offload uses a shared ComponentsManager whose global hook
changes can affect surviving owners. A resident owner and a global model-CPU
owner must finish and release sequentially, even when explicit memory budgets
fit; unsafe overlapping lifetimes are rejected. Genuine group hooks can coexist
with global CPU hooks on the same execution device. Before changing global CPU
hooks, the loaders verify actual group/global hooks and device identity rather
than relying only on policy labels.
Borrowed components preserve their source owner's placement and metadata;
disk-offload file ownership cannot transfer to another loader. A resident
ControlNet remains resident when the pipeline's group recipe does not offload it.

## Native policy replacement

Native loader inspection, component reuse and dispatch share the same bounded
normalization for attention, VAE slicing/tiling and the explicitly selected
inpaint compatibility policy. Missing attention means `inherit`; missing VAE
controls remain unset. An explicit `false` differs from an unset control.
The native inpaint default and `whole_v1` have distinct ownership when the
reviewed Qwen Edit inpainting adapter is selected. Planning inspects these values
without importing model libraries or executing unreviewed suppliers.

Changing a cached native owner's policy destroys its exclusive components and
cached consumers before replacement weights allocate. Shared components and
unrelated owners retain their hooks and placement. Completed consumer media is
detached and retained; opaque execution state is invalidated. Retained media
does not become a cache hit for the changed workflow. If exclusive model
references remain alive, execution stops before replacement allocation.

Auto replans from actual free memory after this release; it retains the existing
capacity, forced-math working-budget and safety-floor checks. A released owner
is not a promise of sufficient RAM or VRAM. Expert uses the same ownership
release after resolving ordinary node arguments, including adoption of a loader
under another document's node ID, without granting Auto memory qualification.

## Compatibility and proof

Direct image/audio pipeline mutation ownership is separate from native Modular
LoRA and Auto memory qualification. Full graph preparation binds the actual
direct loader to one ordered mutation policy, including explicit vanilla state.
Image supports its existing `LoadAdapter`; audio also declares `SetAdapters`
and `FuseAdapters`. Different terminal policies cannot share one mutable owner.
Changed loading controls or changed, removed, reassigned, invalidated or
incomplete mutation state use the same scoped destruction and material-output
retention before replacement
allocation. Unknown connected loading/mutation controls execute only in normal
dispatch and disable warm-owner reuse; unchanged connected recipes also pay this
reload cost while their scalar/configuration suppliers remain cached. No supplier
is run during policy inspection. Existing runtime configuration passthroughs
remain executable with a cold owner; union/alias or undeclared consumers cannot
establish reusable ownership. Loader reuse also requires live registered model
identities, so model-manager eviction invalidates retained dependent references
before the next allocation.
Direct mutations inside loop bodies remain unsupported pending a reviewed
iteration policy. These process-local cache records grant no memory credit and
do not make downstream direct adapters Auto eligible.

Full graph regressions use the real server preparation, executor, NodeBase and
memory ownership with tiny Safetensors and fake upstream image/audio results.
They distinguish warm and prompt-only controls, mutation removal/reassignment,
unrelated owner/media preservation, failed mutation and preallocation release.
This is cache-lifecycle proof, not live Diffusers/PEFT output or hardware fit.

The graph, workflow hash, node identity, precision, prompts, seeds and requested
workload stay unchanged. Selecting the release strategy affects cached model
reuse and load time; scheduled attempts must not advertise reuse of prior-run
owners that the executor clears. No second graph representation or executor is
introduced.

Contract tests cover overlapping unmeasured owners, mixed explicit/unmeasured
demands, shared loader reuse, sequential image retention and loop boundaries.
Executor tests separately assert owner destruction, retained image pixels and
real capacity-check calls using controlled hardware snapshots. Those tests are
not GPU model-fit or visual-quality qualification.
Policy-transition regressions use the ordinary graph/node dispatchers, the
pinned ComponentsManager and tiny CPU weights. They verify destruction before
allocation, shared-hook preservation and rejection when the refreshed RAM
reading is insufficient; synthetic placement labels are not live GPU proof.
The zero-free regression passes a fully occupied XPU reading through the actual
hardware snapshot and next-owner capacity guard; it verifies that a requested
1 GiB allocation is rejected despite a positive fallback reading.
Pre-run CUDA cleanup separately preserves a known zero over a positive fallback.
Genuine tiny CPU components exercise upstream hook ownership, while controlled
snapshots cover mixed-policy release and shared-storage credit. These checks
do not establish live GPU hook coexistence or constrained-device model fit.

A hardware receipt remains tied to its exact models, revisions, workload,
runtime, device and observed behavior. A successful multi-model run on a large
GPU does not establish fit on a smaller GPU. Remaining qualification work should
separate unique weight storage, loading overhead and activation demand without
inventing estimates from repository size or machine tiers. Until those complete
demands are reviewed, Auto preserves the conservative release or unresolved-fit
behavior above.

The 9 October 2026 review counted 37 resource declarations and 80
declared Auto profile/mode pairs, with no complete explicit
`workingMemoryRequirements` budgets. Those counts describe resource contracts,
not the number of templates or qualified workloads. Retained task-scoped memory
captures provide observations for future review; sampled process memory is a
lower bound, and pre-task loading, resident storage and inference peaks are
separate quantities. Custom full-weight cloud runs cannot qualify constrained
Auto or become complete production budgets merely by copying their peak values.

For setup and platform policy, see [Developer setup](developer-setup.md) and
[Runtime support matrix](runtime-support-matrix.md).

## Private ancillary owner boundary

An exact private prompt Preview uses the normal graph task, node-cache lease
and model thread. Successful dispatch records weak process-local owner
observations; workflow hints, matching node IDs and client snapshots cannot issue
release permission. Preview requires explicit Custom memory while incoming Auto
admission remains unqualified. A completed image owner can be associated with its
validated native stage Blocks and ordinary declared stage roles by their exact
role map, typed graph/edge/interface and selected-route identity, plus actual
dispatched cache, code, manager and component incarnations. Presentation, inactive drafts and stale resource scalar snapshots
grant no authority. Unknown cached ownership, shared or active
models and changed worker/task/graph/incarnation refuse before allocation, without
the historical whole-process preparation or failed-attempt cleanup fallback.
Historical graphs without this optional purpose retain their existing behavior.

After the actual provider and proposal dispatch frames unwind, the same task can
release only its own exact registered provider and retaining proposal consumer.
It preflights detached material and the native bounded string summary, preserves
the original proposal producer, and performs the existing scoped weak-reference
checks before publishing completion. A survivor or allocator failure stops after
release; no rollback reload, automatic replay or next image task is submitted.
Failed/cancelled private Preview attempts preserve uncertain ownership for normal
explicit idle recovery. A stale queued claim receives the same preservation.

Before an associated image owner is released, the selected official processor
checks the actual request on CPU. Input-token and per-reference pixel bounds
reject oversized requests without truncation or resizing and preserve the image
owner. The normal executor orders the selected built-in file supplier first;
there is no nested graph executor or shared preprocessed batch. Only after CPU
preflight does scoped release destroy the exact image owner and retaining
consumers, preserving detached media. A per-run hash witness is checked again
before proposal device transfer. Exact dispatched LoRA descriptor ancestors may retain
their detached JSON; other suppliers cannot grant release permission.

The generic image/text loader exposes explicit NF4 quantization for the reviewed
immutable adapter with BF16 compute on primary CUDA. Its existing optional
runtime must be activated normally. The native quantizer, converted language
Linear modules and packed state are checked at load and generation; the vision
encoder and output head stay dense. Parameter storage counts exact tensor
shapes and quantization states; it is not the whole artifact size divided by four.
Raw loader precision and generation defaults remain unchanged.

The normal workflow Auto planner has one finite private T2I NF4 recipe. It
requires the exact immutable configuration, Safetensors header and full 248-Linear
census, the reviewed normal activated runtime, BF16 compute and primary CUDA.
The admitted request is text only: no references or layout, at most 4096 input
tokens and 3584 new tokens, greedy decoding and one beam. The effective registered
enhancement default remains 3584; ordinary HFT registry defaults do not change.
I2I Auto remains closed because its encoded read/decode preparation needs a
separate bounded same-read memory contract.

The full-envelope storage estimate is 12,127,272,928 GPU bytes, with the existing
max(2GiB,10% device total) allocator/vendor workspace policy added separately.
It includes exact dense/NF4 parameter storage, persistent caches/mask, named
attention/GDN/MLP phase lifetimes and current-Linear dequantization. It is a
reviewed estimate, not a mathematical bound on every kernel workspace, measured
physical WDDM residency, or public/live/template qualification. Torch allocation
and reservation counters do not establish physical VRAM availability.

Before provider preparation, actual currently available RAM must cover 256MiB
for processor/tokenizer/input material plus max(4GiB,10% system total), while the
outgoing image owner remains resident. No projected release credit is added.
Only after the actual CPU processor/input witness passes may the exact observed
image owner be released. Actual free RAM and accelerator capacity are then
sampled again before the neural factory; the incoming host estimate is
2,311,061,504 bytes plus the same RAM policy floor. Inadequate capacity stops
execution without a global cleanup, fallback, or implicit image replay.

The actual native loader binds local safetensors, single-device on-the-fly NF4,
Windows pread and disable_mmap=False. The whole-file loading path is excluded.
Fresh reviewed library-source and native-callable checks retain official SDPA
and the reference convolution/GDN paths. At generation, actual module/config/
packed-state checks require the exact census, eval/inference mode, default
dynamic cache and last-token logits, with no histories, alternate algorithms,
kernel replacement, hooks, compilation, offload cache or external model input.
The existing executor and pressure policy remain authoritative.

The diagnostic tensor/KV/vision shape projection is separate and does not open
another recipe. Tiny CPU tests establish arithmetic, configuration, order and
ownership only. Real native image-to-PUP-to-image output, repeated runs,
cancellation and refresh/currentness checks remain required before qualification.

Selected parent validation preserves two narrow ordinary-export compatibility
cases without changing the graph or immutable registered Block definitions.
An optional registry input omitted by the exact selected spec may retain only
its inert `{"display":"input"}` header, with no source/value and no non-null
registry default. The legacy ModelsLoader `inpaint_compatibility` field may
retain only the hidden registry default literal `native` when that exact spec
does not bind it. Connected fields, policy overrides, unknown fields and other
scalar defaults still fail; `False` and `0` are never treated as absent.

The selected ancillary declaration also carries hash-covered independent
verification metadata: pinned source and official classes/version, chat/system
prompt digests, and exact permitted quantization recipes. Provider literals
select runtime dtype/device/quantization; proposal output cannot supply its own
verification expectations. This metadata does not confer model eligibility,
resource admission or Apply permission.

The selected private authoring metadata binds `runtimeInputs` and `requestInputs`
to actual registered literal controls. InitialValues supplies its immutable
provider Hub selection/revision and selected operation. For the T2I private
authoring recipe it visibly selects CUDA, BF16, NF4 and 4096 input tokens through
those same controls; I2I and global registry defaults stay unchanged. Selected `fieldOverrides`
reuse the existing enhancement presentation to expose its effective token limit
and hide the ignored ordinary limit. They cannot change field types, defaults,
options or ranges. Copied parent prompt/layout/media remain parent authoring.

The selected private Preview currently declares `supportedPromptModes` as
`t2i` and `i2i`: its layout must be exactly `None`. A layout object is refused
before queue/preparation, and malformed values are never coerced to `None`.
The ordinary image workflow and authored fields remain unchanged. Raw native
structured enhancement still produces diagnostic output under its strict parser;
it does not prove preservation of existing region IDs. The pinned publisher
structured prompts generate new class/index IDs and a full scene table. A
separate explicit complete-composition proposal requires its own reviewed
contract, parser, atomic UI action and live evidence; it is not enabled here.


### Independent native prompt-proposal cache observations

The existing `GET /cache/{node}/proposal_summary` route continues to return the
raw UTF-8 proposal string. Opt in to `?with_metadata=1` to receive the exact
native string plus an independently captured observation:

```json
{
  "value": "<the unchanged raw proposal_summary string>",
  "observation": {
    "schemaVersion": 1,
    "summarySha256": "sha256:<digest of the exact raw UTF-8 string>",
    "producer": {},
    "dispatch": {},
    "request": {},
    "consumedInput": {}
  }
}
```

The empty objects above stand for the closed producer, dispatch, authored-request
hash/control, and consumed-input observations; they are not valid sample claims.
The original producer is captured before generation. The dispatch is captured
only after the actual native node call succeeds or returns an existing valid
cache output. On a warm task B, `producer` remains original task A and
`dispatch` records B with `status: "cached"`; both semantic graph hashes must
match. A fresh computation has `status: "succeeded"` and a new native output
incarnation even when the returned string bytes happen to be identical. Retry
attempt, session, worker, backend source, and UI correlation values retain their
actual producer/dispatch domains. They are not inferred from summary contents.

`request` contains the exact authored prompt/layout hashes and effective greedy
generation controls. `consumedInput` contains only `schemaVersion`, ordered
`references`, `processedImageGrid`, and `fingerprints`, detached from the actual
CPU pre-generation normalizer/processor result. Rendered text, tensor values,
and media are absent. A reference `sha256` and the legacy
`reference_source_sha256` fingerprints use the existing normalized RGB domain:
the normalizer's JSON geometry/mode header plus RGB pixel bytes. They are not
encoded-file hashes. `Image.Load.source_hash`, when independently observed,
remains the separate encoded-file byte digest.

This option accepts only the exact preloaded native registered string output and
its server-minted detached holder after successful scoped terminal release.
Weak node/output-incarnation observations retain no neural object. Actual cache
deletion, replacement, rename, failure, and release boundaries invalidate old
observations; rename never relabels producer identity. The inventory keeps at
most 64 observations. Missing historical/expired observations require explicit
recompute. A failed terminal release may preserve the original raw diagnostic
string while withholding successful detached metadata.

GET does not execute a node, instantiate a model/processor, mint or refresh an
observation, or mutate cache state. It freshly verifies the current executable
source and static metadata census (at most 1024 directory entries, 256 JSON files,
8 MiB per JSON file and 64 MiB aggregate), exact adapter declaration, and all
reviewed processor metadata bodies, including the external Hub system prompts.
No model-weight body is read by this response path. The JSON envelope explicitly
uses UTF-8 (`ensure_ascii=False`) and `Cache-Control: no-store`. Raw summary and
observation limits are 2 MiB and 32 KiB respectively; escaped envelope bytes are
limited to 4 MiB plus 64 KiB.

Only one `with_metadata=1` option is accepted. An index, another field, duplicate
option, or extra query option returns 400. Missing node/output returns 404;
unsupported/non-native/static output returns 400. Busy, unobserved, replaced,
invalidated, or stale worker/source/cache observations return 409. Without this
option the ordinary cache route retains its existing response behavior.

These observations grant no Apply, release, Auto, artifact readiness, or public
qualification authority. The client must independently bind the current declared
adapter/model controls, authored request/reference expectations, submitted graph
and actual dispatch task/attempt/session, worker/source, and unchanged document
and UI epochs. The current private authoring/Preview declaration supports plain
T2I and I2I; preserving an authored layout is refused before preparation. The raw
HFT structured operation remains available as a strict diagnostic and does not
imply that layout-preserving Apply is supported.
