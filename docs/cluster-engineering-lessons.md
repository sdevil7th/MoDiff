# Cluster engineering lessons and required working procedure

Read this guide completely before work on nodes, Blocks, Modular hierarchy,
graph persistence/execution, model qualification, or cross-machine integration.
It is mirrored in both repositories; update both copies together.

## What went wrong

The repeated Qwen defects were implementation and verification failures, not
unreasonable user expectations. The corrective obligations are:

| Mistake                                                                                          | What should have happened                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Treating registered Clusters, User Nodes and internal containers as different UI products        | Establish one shared Block contract, frame, controls, toolbar, sockets and gestures before extending the catalog. Registration/category are provenance, not a separate behavior system.                                  |
| Claiming containment was fixed after inspecting only an outer rectangle                          | Reproduce resize-before-expand, inspect actual served pixels, and check children against every ancestor's header and connector tray. Cover ordinary leaves as well as containers.                                        |
| Reusing compact width/height as the expanded envelope                                            | Store the user's compact size separately; measure visible descendants bottom-up. Expansion/collapse and sibling collision handling must not rewrite semantic state.                                                      |
| Rendering the upstream hierarchy as decorative wrappers                                          | Use the selected executable path, real typed crossing interfaces and stable endpoints. Inactive branches belong in discovery, not as portless active nodes. Never invent links to make a picture look connected.         |
| Hidden/disabled model controls, blank suggestions, inconsistent ports and missing resize/actions | Inspect a freshly inserted node and an edited/reinserted User Node side-by-side. Explain exact model selection; retain creator suggestions, complete interfaces and normal node actions.                                 |
| Small edits causing reconstruction or authority changes                                          | A prompt/parameter edit changes its own instance value only. Structural edits are atomic, keep compatible fields/wires, and reject unsupported changes with an actionable Fix. No implicit library overwrite.            |
| Reporting completed media without trustworthy settings                                           | Capture submitted graph and actually consumed inputs; distinguish requested controls, resolved inputs, and measured encoded media. Never infer quantization or a seed from a stale form.                                 |
| Fixing old seed bindings by silently changing saved execution                                    | Preserve legacy behavior until explicit repair. New routes bind one Generator initializer to the first actual consumer; non-consuming nodes must not reset its state.                                                    |
| Reporting backend disconnect or stale progress without an actionable failure                     | Preserve task identity, distinguish native worker death from slow inference, surface terminal error/recovery guidance, and test health/history/resource polling during a real run.                                       |
| Testing source that was not the served build                                                     | Freeze paired source identities; rebuild the client, preserve user fields, restart an idle backend when needed, and verify served bytes and process-start source identity.                                               |
| Mistaking harness failures for application failures                                              | Check drag threshold/drop center, focus/viewport, stale Vite modules and independent test caches. Fix the harness transparently; never suppress console errors or weaken geometry assertions.                            |
| Re-running expensive campaigns after every unrelated change                                      | Use an explicit impact matrix: focused red/green tests first, shared gates at a stable checkpoint, then only affected real-model routes. Preserve historical evidence without relabelling it as new.                     |
| Validating an edited tree but executing the original one                                         | Feed the validated composition into the existing executor, bind connected state to its recipe, and prove an added upstream placement executes after User Node save/reinsert. Structural receipts alone are insufficient. |
| Requiring every component in an unpruned hierarchy                                               | Preserve full nested paths, but derive required model components from the selected edited workflow, as upstream workflow-specific loading does. A plain text-to-image edit must not demand inactive ControlNet weights.  |
| Mixing working, tested, release-ready and approved                                               | Track schema, UI, execution, repeatability, visual quality, resource qualification and publication approval separately. A passing image or catalog sweep proves only its recorded scope.                                 |
| Parallel sessions without one integration ledger                                                 | Establish shared bases and owned files, export ordered patches and handoffs, inventory dirty secondary worktrees, and merge source in an isolated staging copy before deployment.                                        |

## Invariants to protect

- One owning `BlockInstanceV2`, one flat effective execution graph and one
  existing backend executor. Nested containment is not a second persisted
  execution system. Ordinary leaves remain ordinary nodes.
- `BlockDefinitionV2` is the reusable baseline. Values and structural changes
  belong to the workflow instance; Save as User Node snapshots the selected
  subtree and its current effective values only on explicit user action.
- Root and internal Blocks share behavior. Fresh root expansion reveals one
  level; further expansion is explicit. Collapse preserves crossing sockets,
  links, parameters and execution scope.
- Palette entries are discoverable by nested category and exact identity.
  Deduplicate identical catalog identities, not distinct user-saved revisions.
  Every active internal leaf/container needs a supported insertion path.
- Official pinned Diffusers behavior is the execution reference. Hierarchy
  wrappers, loop ownership, supported combinations and inactive branches must
  match the selected upstream contract; arbitrary combinations are not promised.
- Compatible model changes retain unchanged prompts, controls, layout and nodes.
  Explicitly describe unsupported family/task changes instead of guessing.
- Never lower creator defaults, weaken type/receipt checks, change prompts to
  hide a failure, accept model terms, bypass guardrails, or publish assets merely
  to make a test green. Use Hugging Face Hub pulls for model/reference downloads.
- Generated catalog identities are rebuilt from combined source contracts.
  Never pick an old hash, fabricate a receipt, or merge minified bundles by hand.
  After deploying a changed source bundle, regenerate dependent inventory hashes
  in dependency order before the final suite. Compare semantic entries, not just
  checksums: a fingerprint refresh must not promote qualification or publication.
- A shared caller parameter is not necessarily a constant throughout execution.
  Follow declared upstream state writers: strength-adjusted step counts and
  image-derived dimensions must reach consumers without being overwritten by
  duplicate literal controls. Test requested and effective values separately;
  preserve existing saved behavior until an explicit, ownership-checked repair.
- An image socket does not establish RGB/channel compatibility. Verify the
  actual deterministic preprocessor output against the consumer contract and
  expose format choices explicitly, preserving old processor defaults.

## Required sequence before a completion claim

1. **Inventory and plan.** Read this guide, architecture and security guidance;
   state the user-visible invariant, exact reproduction, affected families and
   proof needed. Preserve unrelated dirty files. Ask and wait on meaningful
   authority/behavior ambiguity.
2. **Reproduce first.** Obtain a failing focused test or record why a faithful
   reproduction is unavailable. Include the user's exact gesture sequence.
3. **Implement the shared fix.** Keep model-specific adapters small and backend
   owned. Document schema/interface changes in both runtimes with shared fixtures.
4. **Test modifications.** Insert by click and drag; resize before expansion;
   expand/collapse progressively; edit prompts/parameters; add/replace/delete/
   reconnect; Configure Interface; move in/out; Undo/Redo; save/reinsert a nested
   User Node; refresh. Verify unchanged siblings/defaults and precise run scope.
   Test invalid operations and their visible rollback/recovery as well.
5. **Validate the deployed view.** Inspect screenshots at readable zoom; assert
   sibling nonoverlap, every ancestor's content bounds and visible connectors.
   Check the actual field value and step buttons inside scrolling Advanced
   sections, not only the enclosing node rectangle. Exercise refresh while a
   different workflow is running; recovering its progress must not steal the
   user's selected editing tab.
   Confirm the served frontend and running backend match the tested source.
6. **Validate execution where affected.** Submit through the frontend, capture
   exact inputs/model revision/dtype/quantization/offload/seed and output hashes,
   inspect media quality, and distinguish export success from visual success.
   Do not restart a worker or resubmit a job while its outcome is unknown.
7. **Close the checkpoint honestly.** Record commands, failures/skips, source and
   hardware identities, preserved assets, remaining defects and next work.
   Never turn a technical success into user approval or Auto qualification.

## Cross-machine preservation and merge

Before retiring a machine, preserve code history, all secondary worktrees,
uncommitted changes, generated media, reference inputs, saved workflows/User
Nodes, run receipts, exact parameters, test evidence and session handoffs.
Keep a private destination outside public Git; do not transfer credentials.
Model weights may be excluded only with a recorded scope decision; retain exact
revisions and file inventories for later Hub pulls.

Checksum the transfer, validate patch hashes and retained asset/sidecar links,
and keep immutable before-copies. Resolve semantics as well as textual conflicts.
Rebuild generated catalogs and frontend assets from the unified source, preserving
backend-owned user fields and local Gallery installs. Hardware-specific receipts
remain tied to the machine and source that produced them.

A completed copy is not a completed merge; a completed merge is not proof that
every model works. Provider teardown needs a verified backup and explicit
authority. Stopping jobs or a service does not establish that billing has stopped.

## Shared-memory accelerator qualification

CPU offload is not automatically a lower-total-memory strategy on a shared-memory
GPU. Track host RAM, swap and accelerator allocations together. Qualify explicit
resident/group-offload alternatives with the same prompt, resolution, steps,
precision and model revisions; never silently change creator defaults or relabel
the failed resource recipe as passing. A long first step with rising RAM/swap is
not evidence of harmless compilation. Inspect pressure promptly, distinguish a
kernel OOM kill from an assumed driver failure, and verify supervisor/UI recovery
before retrying a known failed or cancelled task.

## Native edits and dynamic input types

A successful run seeded through numeric test data does not prove the same field
works after a user edits it. Exercise the real input, commit focus, Save/refresh,
and execute the resulting persisted form value. Numeric widgets can persist
formatted strings; dynamic reviewed adapters that skip whole-schema casting
must normalize them against the exact upstream input declaration before assigning
Pipeline State. Cover scalar-or-list controls, zero, booleans, invalid values,
and text/opaque unions that must not be guessed. Capture normalized consumed
values without rewriting saved instances or replacing immutable defaults.
Do not patch a test to inject a number to hide a native-edit runtime failure.

## Compact receipts and proof capture

Compact transport must preserve identity semantics: an execution runtime hash
and a resource-cache hash are not interchangeable even when both start with
`sha256:`. Test a full websocket completion against its compact queue/history
counterpart with both hashes present and deliberately different. Missing execution
identity stays missing; do not weaken proof validation to accept a mismatch.
Capture runners must cover their fresh-run, resume and error branches. A successful
image followed by a capture exception is retained diagnostic evidence, not a
qualified run. Check unresolved variables in executable scripts as well as typed
application code, and retain expensive output/receipts before stricter validation.

An edit node can contain both input media players and generated-output previews.
Never capture the first `audio`, `video` or `img` element as execution evidence.
Select the declared output preview, match its execution to the backend receipt,
and archive the original media using that receipt's hash; playback previews may
be transcoded. If capture selected an input, retain the failure, recover the
existing output by exact task/hash and repeat only the missing UI assertion.
For User Node reinsertion, compare resolved defaults plus instance overrides and
the exported execution graph—not the overrides map alone. A newly inserted User
Node correctly inherits its saved settings with an empty override map.

A second Run can reuse cached outputs. Record it as cache-reuse coverage unless
compute caches were cleared through the UI or a fresh instance performed actual
inference. Retain task timings and compare output hashes across independent runs.
For template acceptance, await the fresh-session task chooser or the restored
active-run workflow; open an explicit new tab and never edit the running graph.
Then verify Run
readiness again after Save/refresh—not just exported values. Registry hidden
defaults must not overwrite persisted mode-specific visibility or required inputs.
Contract comparisons must use the same effective defaults as execution requests.

## Progress and interactive validation

An edited full hierarchy must preserve selected-workflow specializations, not
only Python class names. Compare every retained placement contract across the
family and test required inputs against the real upstream implementation.
Run, API export and portable workflow export must carry the same execution
recipe; compare exported nodes and paths with a successful actual submission.
Freeze the frontend or disable HMR for final browser regression runs. Editing
served modules during a suite invalidates its reliability as release evidence.

Exercise deletion after an actual loaded-model run, not only on an empty cache.
Large node destructors must run off the HTTP event loop. Serialize cache teardown
with graph execution and field-action creation so a same-id loader cannot reuse
a component collection while its previous owner is being destroyed. Request
cancellation does not stop a worker thread: retain the ownership guard until
cleanup finishes. Test responsiveness, replacement ordering and cancellation;
do not hide a stalled server behind a longer export timeout.
Cache deletion must enumerate removed nodes from the durable effective graph,
including hidden descendants of collapsed Blocks. Conversely, hiding a retained
projection must not evict its executable node. Test both cases and unrelated
cache owners, then verify native deletion against a genuinely loaded model.
Off-thread execution alone does not make teardown memory-safe. The pinned
ComponentsManager's per-component removal rebuilds hooks or copies a model to
CPU. For destruction, remove the selected ownership records and only unshared
components without offloading; retain shared components and surviving hooks.
Test this adapter against the actual pinned manager, including peer-hook cleanup,
and re-review it when upgrading Diffusers.

Profile a generation-time navigation or containment timeout before increasing
waits or changing layout. High-frequency progress timestamps are not changes to
graph eligibility. Keep readiness invalidation tied to the task lifecycle,
ownership and labels it actually reads; resource and graph validation keep their
own signals. Test both a burst of progress-only updates and each meaningful
status/ownership change. Preserve progress reporting, strict graph validation and
the original geometry assertions, then repeat the concurrent generation gesture.

## Audio execution and delivery

- Inspecting a Fix proposal is not testing Apply Fix. Exercise Apply, Undo/Redo,
  reload and execution export. Internal repairs must update the owning Block's
  effective graph through the same connection rules as native gestures.
- A collapsed/expanded socket alias must retain the bound input value. Do not
  diagnose an empty generated alias independently of the owning instance.
- Do not propose cross-boundary repair wiring that the canvas rejects. Internal
  source/bridge additions require explicit adoption or interface configuration.
- Native cancellation can replace a model worker. Test durable cancelled status,
  reconnection and a fresh unchanged-settings run; bound capture HTTP timeouts.
- Inspect inactive controls as well as active values. Seed exact registered
  adapter field contracts without changing defaults or authorizing execution.
  Same socket type does not make a music control meaningful for a sound model.
- Preserve native frame-quantized audio lengths. Verify the upstream formula
  before changing duration tolerances; never pad the output to satisfy a test.
- Static template inventory imports must be offline and side-effect-free even
  after bundle chunk changes. Use ephemeral browser shims and reject network
  access; regenerate dependent inventories after the final deployed build.
- A completed backend task without a delivered preview is not a passed UI run.
  Retain failed delivery diagnostics and require exact-task browser playback;
  do not replace that assertion with an HTTP-only download or a longer timeout.
- Profile progress handling before adding timing slack. Immutable durable-node
  references can keep policy, persistence and telemetry routing stable while
  progress changes; prompts, params, ownership, layout and unknown fields must
  still invalidate. Do not cache schema validity across mutable input records.
  Isolate socket writers so a stalled browser cannot block another client's
  completion. Assert durable gesture results, not an expiring success snackbar.

## Reviewed value-port changes

- Test the public static registry, not only the Python class's `params`.
  Schema constants must be exported by the package for the AST registry parser.
  Otherwise palette nodes and compiled Blocks can have different sockets.
- Regenerate pinned definitions only after the registry test passes. Verify the
  declared socket in both the served registry and a newly compiled instance.
- Inherited state inputs must not acquire registry defaults. Explicit input and
  output names need collision-safe identities; loop descriptors are not tensors.
- Freeze application source before browser gates. HMR during a test invalidates
  that run; rerun against the final source, not just the isolated failed assertion.
- Measure every visible socket against its actual node bounds and hit-test the
  native connection gesture. Intrinsic grid tracks can overflow even when the
  node and ancestor rectangles themselves pass containment checks.
- Test Arrange immediately after save/reinsert or focus. A temporary automatic
  fit suppression must not swallow the user's explicit Arrange action.

## Modular authoring and persistence

- A shared Text Inputs test repeated across routes does not cover their distinct
  placements. Test every executable placement and the actual loop/decode classes.
  The wider Qwen matrix found 51 Edit Plus/Layered source-lookup failures hidden
  by the shared-step test: flattened catalog names and nested runtime paths differ.
- A moved node retains identity even if a new node occupies its previous path.
  Test successive Save/reuse cycles; do not infer original identity from position.
- Declared component mismatches are advisory for editable drafts. Execution must
  check actual connected components, accepting valid subclasses and preserving
  exact source authority. Do not forbid a valid user replacement based on the
  original loader declaration.
- Iteration links must resolve to one concrete loop instance, not merely equal
  path strings. Test native pointer wiring, carried-state initialization and
  actual multi-iteration upstream output, including through the runtime adapter.
- Leave unbound loops on the existing path. A new optional adapter must not
  inspect or mutate members when the feature is unused. Preserve OOM categories.
- Fix must retain understandable manual-only diagnostics. No automatic candidate
  is not the same as no issue. Test that the dialog actually displays the error.
- Updating a reparenting reducer does not connect the native drag handler to it.
  Verify the complete pointer path; a layout-only guard can bypass correct logic.
- Runtime-only failures need the same visible correction path, restricted to
  their exact active workflow/run. Keep them advisory after edits until a new
  execution verifies the correction; never fabricate an initial tensor or model.
- Diagnose the effective bound iteration input, not only the enclosing loop's
  original Pipeline State. Report bounded type/shape summaries, never contents,
  and preserve the resource-error category for memory failures.
- Inspect the complete Fix dialog, not just the expected sentence. Collapsed
  Blocks can already contain connected outputs; do not offer an unrelated new
  Preview because its projected child is hidden. Keep disconnected-output repair.
- Repeated passive failure events must not close a dialog the user opened or
  replace its selected task with an unrelated background failure.
- Compare palette-inserted loop members with compiled ones. Do not expose legacy
  ordinary controls that only step execution consumes; explicit iteration inputs
  must be the controls offered on new loop recipes. Preserve saved snapshots.
- Automatic DOM minimum-size measurements are not user edits. Test Undo followed
  by the queued measurement and then Redo; recording the measurement as a resize
  can silently destroy Redo. Manual resize gestures must remain undoable.
- Test targeted Fix with unrelated unresolved prerequisites, including an absent
  large model. Users need to skip individual suggestions and repair their draft
  without installing anything or changing another part of the workflow.
- Moving a catalog route from standard fallback to native Modular execution does
  not authorize retiring saved loaders' explicit execution-profile identity.
  Keep historical loaders resolvable and non-public; test new admission selection
  separately so compatibility does not create duplicate catalog choices.
- Explicit Save and autosave must share a per-document write chain. An own-save
  acknowledgement is metadata, not a remote edit: a delayed receipt must not
  replace a newer clean document restored by Redo. Test out-of-order responses,
  exact dirty clearing, and genuine remote updates separately.
- Native drag evidence must check the actual hit target. A floating activity
  notification can cover a node and open a past workflow instead; dismiss local
  notifications without deleting history. XYFlow establishes its grab offset
  after the drag threshold, so cross that threshold before measuring a synthetic
  long drag. Confirm the intended node-center destination before mouse release.
- Image-to-image denoising is not instruction editing. Use a descriptive target
  caption for FLUX.1 img2img, not a Kontext "change only" prompt, and assess image
  preservation against the actual strength. A successful tensor run can still
  fail the requested visual change; record that failure and retain its evidence.
- New backend-defined catalog routes must be inserted through the real palette
  and compared with their compiled creator defaults. A missing legacy frontend
  model profile must not import another model's prompt, size, steps or guidance.
  Verify submitted values before calling a run full-setting; preserve old instances.
- Direct `execute()` tests bypass normal NodeBase dispatch validation and caching.
  Include actual graph dispatch for reusable ordinary nodes. Invalid selector
  choices must report their field before execution, never become empty lists or
  dictionaries that fail later inside tensor/model code.
- A selected contract stored in a hidden signal does not expose its controls.
  Compiled ordinary Blocks that skip dynamic field actions must initialize the
  selected field schema too. Verify a real visible socket and native connection,
  not merely the contract JSON or a direct Python call.
- Native projected-leaf wiring must use the same scalar-type aliases as V2
  compilation and public sockets. A passing compiler is not evidence that the
  canvas accepts the connection; drag a real ordinary value node into a Block.
- Test the preferred pipeline offload path, not only its component fallback.
  VAEs called through encode/decode need leaf hooks. FLUX.2 reads BatchNorm
  statistics outside forward; disk hooks can replace those buffers with empty
  storage and return black images. Preserve direct-read buffers, assert their
  values, and inspect decoded pixels before reporting a model pass.
- A mutable Generator cannot be reused from node cache across unchanged runs.
  A seeded producer must create a fresh object per graph execution, while
  preserving intentional sharing among consumers within that execution.
- Test discovery through the actual client parser after adding a model profile.
  A previously ignored model can expose inconsistent backend metadata. Public
  optional-runtime aggregates must enumerate exactly the published execution
  profiles; hidden compatibility loaders remain selectable by saved identity,
  not extra invisible requirements of new admissions. Keep strict client checks.
- Component download completion and exact workflow install readiness differ.
  Validate the complete reviewed file selection and its persisted download plan,
  including small metadata files, instead of bypassing a Model Manager Repair
  state merely because weight files exist. Reuse cached bytes through Hub.
- Authorized standalone-weight cleanup must also reconcile its persisted
  download-plan selection, counts and byte totals, retaining the old plan and
  exact revision. Validate every retained file/shard first and then check public
  model readiness/fingerprints; a successful generation alone is insufficient.
  Leaving the removed standalone entry required caused false Repair states and
  blocked valid template provenance. Never solve this by ignoring arbitrary
  missing files or weakening the shared completeness validator.

## Factory components and optional-runtime gates

An upstream AutoProcessor/AutoTokenizer declaration is a factory, not a runtime
base class. Validate pinned artifact classes separately from the concrete object
returned by the genuine installed factory. Test the loaded block boundary as well
as model-index parsing; passing the latter alone can still leave generation broken.

Run relevant contract/signature tests with the app's approved optional runtime
activated. A baseline skip is not a pass. Keep both results: activation can reveal
stale fixtures or missing capability assertions hidden by the baseline environment.

Supervisor tests require isolated filesystem state as well as isolated ports.
A port-zero or mocked listener can still reconcile the real durable queue at
startup. Patch the configured data directory to a temporary directory before
constructing any real controller, and assert the exact queue path. Otherwise a
parallel unit suite can produce false worker-crash notices during real inference.
Retain the affected generation; repeat only missing UI assertions after fixing
test isolation, without relabelling it as a fresh inference pass.

## Offload file ownership and executable provenance

Disk group files are weights, not reusable scratch keyed only by a node name.
The pinned upstream pipeline helper shares a directory between components;
group filenames use internal group names and existing files are reused. Allocate
distinct storage per component AND new hook owner/load. Preserve the existing
hook directory only for that same live owner. Test equal-shaped wrong-weight
substitution as well as shape errors, same-node model replacement and repeats
against direct upstream GPU tensors. Include auxiliary ControlNets/encoders and
VAE direct encode/decode entry points. A plausible image cannot detect wrong
weights; retain but invalidate affected old offload proofs until retested.

Output provenance must identify the actual loader's model independently of any
downstream decoder and auxiliary adapter. Scope adapter fields to the adapter
node, never let its revision overwrite the base-model revision, and assert model,
prompt and dimensions in native generation tests, not merely a completed status.

## Native interaction and visible control acceptance

Test editable-body height as well as outer node/ancestor bounds. A correctly
contained node can still have a 20px body after its ports consume a fixed height.
Reserve the shared header/toolbar/tray and a usable scroll viewport in projection;
feed those dimensions into sibling placement and ancestor containment. Exercise
actual internal text editing and saved-value locality, not only collapsed fields.

Qualify fixed resource recipes with an explicit native Expert selection. Do not
depend on Auto coincidentally selecting the same recipe or silently apply a
different plan. Rejected preparation must surface from every Run entry point,
including the selected-node toolbar, without an unhandled promise or submission.

Distinguish built-in data-operation identifiers from model identities in output
receipts. A JSON producer's pipeline_class is not necessarily a model. Likewise,
test free-form model selection through typing, Tab/blur and an exact repository
assertion before allocating weights; a successful combobox fill is not enough.

Test random generated identities against persistence with deterministic boundary
cases. A valid nanoid may end in an underscore; filename slug sanitation must
not trim immutable User Node IDs or alias neighbouring records. An error that
mentions permitted characters/length is not proof that a model's long name
caused the failure. Inspect the generator and exact validator before diagnosing.

Keep backend input capture and the client's strict receipt parser in lockstep.
Generate contract fixtures from the backend's complete allowlist, including
node-scoped auxiliary revisions, scalar lists and omitted opaque values. A valid
new receipt rejected by an old parser must not be mistaken for uncaptured legacy
data. Verify the actual Gallery inspector before and after refresh.

Production static, dynamic and preload imports must use the same versioned
module URL. Minification can change quotes to backticks; missing those references
loads a second state-store instance even when its hydrated JSON looks identical.
Test the actual emitted bundle and browser resource URLs, not only the dev
server. Fix module identity instead of weakening graph-change guards to mask
the duplicate stores. Preserve rejection of real concurrent workflow edits.
Control-area assertions must include the shared Block frame as well as ordinary
node bodies; an empty result from a renderer-specific selector proves nothing
about missing controls.

Cold-start the actual production lazy panels with their persisted visibility.
An icon descriptor initialized at module scope can capture an undefined export
when chunk splitting creates an entry/panel cycle. Keep shared icon dependencies
in the vendor chunk; test cold startup and tab icons, not only already-warm panels.
Document and bound any measured bundle-size cost of that correctness fix.

Inspect the generated region, not just global image contrast. An unchanged
outpaint canvas can contain a detailed source photograph and still leave every
new border black. Compare controlled strength changes without changing prompts,
steps or precision; preserve failed evidence and existing saved settings when
correcting new-form recommendations. Assert each extension separately.

Validate creator parameter values against every related action's runtime bounds.
A control T2I adapter accepting guidance 30 does not prove its edit/inpaint
siblings accept the same reviewed workload. Preserve exact old presentation
contracts when widening a justified bound; do not accept unrelated schema drift.

Exercise generated-ID edge cases deterministically. NanoID may begin with an
underscore or hyphen, while Block semantic and instance IDs require an
alphanumeric prefix. Use valid prefixes for new ordinary nodes and duplicates;
explicit adoption of old IDs must update crossing edges atomically and retain
Undo. Test both punctuation prefixes, not just whichever random ID a run emits.

Run the relevant installed optional-runtime gate, not only the base-environment
gate that skips those tests. Compare newly found failures with the unchanged
baseline before attributing them to a patch. Promoting a pipeline out of a
contract-only registry must preserve exact discovery-generator coverage and
aliases. Reproduce the existing snapshots byte-for-byte; do not regenerate
different identities merely to make a stale generator pass. Typed-component
tests must use genuine compatible component types, without weakening runtime
checks to accommodate an obsolete dummy fixture.

Do not import a sealed optional-runtime directory through bare PYTHONPATH for
tests. That can write unowned bytecode caches and correctly make the next app
startup refuse the runtime. Use the backend's validated optional-runtime test
entry point with bytecode writes disabled, including subprocesses. If test
caches contaminate it, preserve them in quarantine and verify every original
locked byte; never relax the integrity boundary or silently reseal new files.

Poll lightweight queue status during inference and fetch full run history for
terminal provenance or explicit inspection. Before another GPU run or cleanup,
check both the active task and the actual public `queued` map; an invented
`pending` field does not establish that the queue is empty. Regression-test a
waiting task with no current task. Repeated full-history reads contend
with Gallery commits and do not model the normal UI progress path. Keep a strict
status latency assertion and retain failures; changing polling must not turn a
failed original lifecycle into a pass. Recover exact completed tasks separately,
then rerun the affected native lifecycle. Keep test sources stable during a run
so failure locations and the preserved test version remain attributable.

A validated optional-runtime test launch describes an active process. Unit tests
that fabricate base-worker catalogs must explicitly isolate that process-status
environment, restoring it after each case. Do not clear the active status in the
shared runner or relax production mismatch checks to make those fixtures pass.

Run coverage-ledger verification after installing the final frontend build.
Rebuilding a shared dependency can change the template-bundle byte fingerprint
without changing any template. Regenerate from exact reviewed upstream sources
and compare the semantic inventory; do not blindly replace a hash or change
qualification status. Retain the failed pre-update gate and rerun the full gate.
The dependency order is upstream coverage → template candidate contracts →
template authoring specs → Comfy contract resolution and Comfy evidence resolution.
Use their existing generators; check that hash-only refreshes leave every other
JSON value unchanged before accepting the coordinated ledger update.

Outpaint is not one universal graph recipe: some admissions take a prepared
source canvas and mask; others create them internally. Validate the exact
admission's input geometry and mask before inference. Both paths must still prove
that the generated extension is filled and assess seams separately from execution.

Moving a concrete step out of an upstream conditional group is not proof that
the original group's selector can still run. The explicit MoDiff graph executes
the selected reviewed steps; component initialization must not re-select the
old branch to infer demand. Install applicable connected components and validate
each executing step's real requirements. Test initialization as well as the
moved step: pre-issued fake Pipeline State can conceal an initialization failure.

A composition inspector must acquire its own pinned catalog and hierarchy;
do not assume the palette was opened in the current page session. Test opening
it after refresh with the palette closed, metadata failure and an explicit
retry. Loading or retrying metadata must not modify the workflow or imply that
structural inspection has executed or qualified a model.

## Connected internal ports and explicit movement

A workflow crossing edge uses the existing Edge document with an exact semantic
endpoint encoded in a root handle (`block-crossing:input|output:<encoded node>:<encoded field>`).
This is a workflow connection, not a new declared reusable port. Resolve it to
the visible leaf or nearest collapsed ancestor for rendering and to the same
leaf for execution. Removing the last connection removes the derived socket.
Reusable saving retains only contained nodes and deliberately declared ports.

Run Block isolates the selected containment subtree before resolving external
dependencies, including dependencies through existing declared inputs. Use stored
fallbacks or report missing inputs. Whole-graph execution retains crossing edges.
Test all enabled local terminal branches, not only the first preview.

Plain movement changes presentation and grows containing frames. Only the
explicit modifier drag or toolbar move changes ownership. Moving a node must
transfer or prune its old local control/port/preview bindings, preserve crossing
connections, and leave the immutable source definition intact. A removed source
preview remains in that immutable snapshot but is absent from the instance's
active preview inventory. Both runtimes validate that inventory.

For native browser evidence, use the application's read-only E2E export hook
for state inspection. After Vite HMR, dynamically importing a bare store URL
can instantiate a second store beside the versioned module used by the canvas.
A visible wire and a stale test snapshot are not contradictory application
states. Do not mutate stores to conceal a failed native gesture; retain the
original failure and verify actual pointer targets and the durable export.

Template startup must await current backend execution specifications before creating
a managed skeleton. Protect the template-building transition from custom-graph Auto
detachment. Await the loader's backend-published component identities, then wire
stable component inputs before requesting dynamic schemas. An exact specification
with missing dynamic fields returns no desired edges; preserve existing component
wires while it is pending instead of treating that as a request to delete them.
Deleting those wires clears staged input signals and prevents consumer schemas
from arriving. Missing fields must continue to block execution. If a late update invalidates a completed graph receipt, schema-stability
waiting must invoke the existing finalizer again, not just poll the invalid proof.
Require declared template adapters to exist and be connected in native regression
evidence; a base graph that omitted its LoRA is not a complete template.

Generated previews need exact retained media references. Test reopening after a
backend restart and compare the decoded image URL and bytes with the recorded task,
not just `naturalWidth > 0`, which can also accept a placeholder. A same-workflow
backend preview slot may restore missed completion; unrelated recent outputs may not.

## Batch membership edits and workflow resource assessment

Move only the highest selected nodes in the containment tree. Preflight every
destination, then reuse the established adoption/detachment reducers under one
history transaction. A partial failure must restore the entire selection, edges
and effective definitions. History restoration can produce canonical roots only;
rematerialize their projections before the next gesture or after Escape. Ordinary
multiple-node dragging grows every affected containing frame without reparenting.

Nested-container crossing wires need not be declared reusable ports. Both schema
validators must validate the actual leaf field and direction while accepting the
undeclared crossing. Rendering derives connected-only sockets; reusable interfaces
and immutable source definitions remain unchanged.

Workflow resource assessment expands the same requested scope as execution. It
counts actual model loaders and adapters, inspects downstream consumers, and uses
the existing backend batch planner only where an exact source form is available.
Bind results to semantic graph/settings and the active tab, cancel stale requests,
and exclude positions, collapse state and runtime progress from their identity.
Individual recipe estimates do not establish combined memory lifetimes or grant
Auto execution authority. Keep unsupported custom scopes explicit and Expert
execution available. Assessment must not execute models, randomize seeds, install
runtimes or rewrite graph inputs.

## Workflow Auto and explicit legacy membership moves

- Plan the exported execution scope after composition lowering. Reuse backend
  execution profiles, artifact checks and resource candidates; do not infer model
  ownership from a Block label or use one source receipt for several edited Blocks.
- Count shared loader identities once and independent loaders separately. Include
  connected adapters and loop-member workloads. Budget retained owners together
  unless eviction is an implemented and tested executor behavior.
- Keep inspection read-only, including random seed fields. Bind Run to the exact
  planned API graph and revalidate resources at dispatch before node execution.
- Adapt only legacy instances touched by an explicit membership move. Use their
  current embedded/materialized graph and the established snapshot adapter. Keep
  conversion and the complete batch move inside one Undo/rollback boundary. Do
  not save library copies during movement or replace current values with defaults.
- Test legacy NanoID prefixes, public wires, completed previews, mixed selections,
  missing destinations and Undo/Redo. Native evidence must use actual selection
  and drag gestures; a reducer test alone does not prove toolbar behavior.

## Auto owner lifetimes and computed controls

- Keep the existing executor. Only budget released memory after its ownership is
  removed, shared references are accounted for, and downstream material outputs
  are detached. Check actual free memory before the next model allocation.
- Inspection never executes resource suppliers. Resolve pure built-in operations
  or mark preparation pending; execute reviewed data-only ancestors once at Run,
  invalidate old caches, replan, and compare the actual consumer arguments.
- Apply deferred offload changes through the existing resource event and owning
  workflow transaction. Preserve newer edits and verify complete shared-control
  groups before changing loader settings. Record actual values and releases.
- Reuse nested legacy snapshots and exact Cluster metadata. Convert the touched
  workflow instance atomically with movement or Auto changes; retain current
  values, hierarchy, crossing wires and previews without changing the library.
- Prove cache destruction and material-output retention in executor tests, then
  retain a real multi-owner model run separately from contract-test evidence.
