# Workflow Auto memory contracts

Workflow Auto plans the exported execution graph and runs it through the existing
backend executor. Shared loaders count once; distinct loader identities count as
independent model owners, even when they select the same repository. Planning
never loads models, installs packages, changes prompts or reduces creative
settings to make a workflow fit.

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

## Compatibility and proof

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

The executable inventory currently contains 37 resource declarations and 80
declared Auto profile/mode pairs, with no complete explicit
`workingMemoryRequirements` budgets. Those counts describe resource contracts,
not the number of templates or qualified workloads. Retained task-scoped memory
captures provide observations for future review; sampled process memory is a
lower bound, and pre-task loading, resident storage and inference peaks are
separate quantities. Custom full-weight cloud runs cannot qualify constrained
Auto or become complete production budgets merely by copying their peak values.

For setup and platform policy, see [Developer setup](developer-setup.md) and
[Runtime support matrix](runtime-support-matrix.md).
