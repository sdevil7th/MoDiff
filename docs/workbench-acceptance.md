# Workbench acceptance and qualification boundaries

This guide consolidates the durable requirements of the retired workbench,
workspace, image-readiness and integration plans. It is an acceptance standard,
not a new implementation plan or a claim that every route has passed.
Dated checklists, run IDs, machine inventories and abandoned proposals remain
recoverable in Git history. Keep this guide identical in both repositories.

## Current contract, not historical UX

Use the current workflow-authoring guide and custom-node documentation for
product behavior. The old Creator/Developer audience switch, separate approval
wizard and destructive Cluster-to-User conversion are superseded. Memory
Automatic/Custom is independent of authoring. One backend graph executor,
resource owner and task-generic node contract remain authoritative.
Model/library-specific behavior belongs in reviewed backend adapters.

Native stage support means real editable upstream stages, not several decorative
nodes around one pipeline call. Standard whole-pipeline adapters remain valid
when the exact upstream task has no usable native stage chain. Document each
exception with artifact/task/revision, upstream evidence, supported controls,
unavailable stage editing/reuse, transition behavior and a revisit trigger.
A missing MoDiff adapter is not an upstream exception. Family resemblance and
nominal tensor types do not establish semantic compatibility.

Model/task edits must be atomic and undoable. Preserve compatible prompts,
authored settings (including values equal to old defaults), custom nodes,
connections, layout and explicit interface removals. Keep unsupported values
recoverable, explain incompatible roles and require review for destructive
changes. Resolve compatibility from declared roles, conditioning/latent formats,
lineage and capabilities; unresolved compatibility is not guaranteed compatibility.

## Freeze the qualification denominator

For a broad image release, enumerate the union of advertised routes from
operation contracts, starters, model pickers, templates, execution profiles,
custom examples and supported-model documentation. Cross-check exact native
upstream exports independently of the current adapters. Freeze a snapshot/hash;
advertisement or execution-contract changes reopen affected rows.

Each row identifies the immutable artifact revision, task, implementation,
adapter/component variant, advertisement entry points, installed dependencies,
required inputs/controls, native-stage or whole-pipeline disposition, resource
policy, exceptions and evidence. Keep contract-only inventory distinct from
advertised runnable routes. Do not hide missing integrations to obtain a pass.

Separate structural, automated, real-browser, actual-generation, media-quality,
reuse/resource, service-export and platform evidence. Generated inventory hashes
and route counts do not promote live execution or public asset approval.
A bounded installed-model Windows test is not an all-model release gate.

## Required integrated journeys

- Create fresh text-to-image, image-to-image and edit workflows. Required media
  starts in a separate connected loader with a working preview.
- Encode Inputs retains ordinary node layout and resizing. Supported optional
  image sockets stay connectable; text and image encoding can coexist.
- Connect compatible sources directly. Preview output can feed a separate
  downstream branch or saver; an upstream cycle is rejected with a clear toast.
- Switch models/tasks and back with edited prompts, custom nodes and wires.
  Test distinct semantic/topology transitions, not only aliases of one family.
- Change guidance, remove interface rows, save/reopen, refresh, Undo/Redo and
  reorder workflow tabs; compare the effective graph and actual consumed inputs.
- Exercise reviewed local discovery, Python-file drop, import errors,
  registration, compatible suggestions, disable/enable and edited-source Reload.
  Discovery and workflow import never grant execution consent.
- For supported audio, verify separate loaders, decoded input previews,
  generated playback, task-appropriate duration/rate/channels and valid wires.
- Run an exported service from an edited graph with changed prompt/seed/files
  and verify named outputs and declared code/runtime requirements.

For each claimed artifact/task, prove a cold baseline and applicable unchanged,
prompt, seed, guidance/steps, size/batch/reference, persistence, recompute,
offload/release and failure/cancel/retry cases. Record consumed values and
load/encode/reuse observations. Use task-specific equivalents for analysis,
upscale, unconditional and other non-prompt workflows; justify N/A explicitly.
Representative transition tests require a coverage mapping and do not replace
each advertised artifact's execution evidence.

## Outstanding scope is not erased by documentation cleanup

Historical plans left broad inventory/native-stage parity, all-model variants,
adapter/control compositions, service/demo handoff and cross-platform resource
qualification open or partial. Reconcile these against current source and
receipts before claiming closure; old unchecked rows are not automatically
current defects, and old checked rows are not current qualification.

In particular, re-audit advertised native variants (including previously
equivalent-only ERNIE routes), quantized component bindings, conditioned/edit
routes, repeated-run reuse and final continuous custom-node/model-switch
journeys. Retain exact per-model blockers and whole-pipeline exceptions.
Windows Qwen on a 16 GB VRAM / 32 GB RAM host remains unqualified unless an
appropriate exact-route receipt proves otherwise; other machines' results
cannot close that boundary.

## Bounded execution, trust and publication

Use one workflow at a time with an explicit wall-clock bound and cancellation
plan. Preserve the task identity, failure and cleanup result; do not resubmit an
unknown active run. Inspect an unchanged recipe's failure before retrying.
Research quality recipes against the exact official source. After two quality
failures, require a changed engineering/recipe hypothesis or explicit exception.

Never infer permission to download weights, install optional runtimes, execute
remote code, delete cached models, publish assets or commit from a request to
test. Review immutable custom sources and dependency hashes. Keep approvals
outside source packages. Before any separately authorized cache deletion,
enumerate exact-revision dependencies and preserve outputs/receipts; verify
queue state and the app's current capacity/download plan before installations.

Validate generated bytes and media behavior separately from aesthetic quality:
decoded dimensions/count or audio duration/rate/channels, finite/nonempty output,
task fidelity and visible/listening assessment. A short technical audio smoke
is not music-showcase approval; hashes establish identity, not quality.
Public media needs rights/provenance review and immutable Dataset/hash metadata,
not generated binaries in Git. Auto and publication eligibility require their
own exact-route authority and evidence.

## Proof and handoff

Record paired commit IDs plus dirty-source identity, process-start identity,
served bundle hashes, exact models/runtimes, steps, expected/actual result,
proof level and screenshot/log/receipt locations. Preserve failures and skips.
Reproduce suspected regressions on the baseline where feasible.

Use focused failing/passing regressions while iterating, then the applicable
CONTRIBUTING gates at a stable candidate. Reuse unchanged evidence with its
original scope; documentation-only cleanup does not require more inference.
Retest changed execution contracts and validate emitted production assets.
Do not call a handoff complete while required integrated gates remain failed,
unrun or blocked. Report those limits and leave commit/publication decisions
to the maintainer.
