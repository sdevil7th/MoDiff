# Mellon comparison follow-up and remaining feedback plan

This updates the work order after the user reviewed the Mellon comparison.
R01 Run recovery, R02 selected-workflow setup and the bounded R03 prototyping
journey remain delivered. The user accepted that R03 journey as simple enough
for quick prototyping. Supplemental checks below do not reopen that verdict or
claim complete Mellon parity. R04–R06 remain explicit workstreams.

The comparison inspected Mellon backend
`5fd242921d13bff9fb03f4de405fdd39c2335e1f` and client
`af0c5801f843453a1700733596e99fe6589b2e86` read-only. Node counts alone do not
establish editing effort, developer usability or runtime correctness.

## Work order and completion criteria

| Order | Scope | Current state and required outcome |
| --- | --- | --- |
| 1 | R03-C custom-code developer loop | Bounded native text-processor loop checked, with the limits below. Ordinary import, typed discovery, connection, execution, installed-source edit, explicit Reload and saved-workflow reopen worked without rebuilding. |
| 2 | R03-D stage-editing discoverability | Implemented; exact validation and delivery checkpoints are recorded in the same paired PR descriptions. Visible Edit stages and Expand block actions reuse existing transactions. Saved Blocks and Python code have distinct creation/save descriptions. Original browser RED and focused GREEN are retained. |
| 3 | R03-T same-model task continuity | Bounded editor acceptance verified without a product change. Same-model Klein modular text-to-image/image-conditioned round trip preserves authored controls, source wire, Resize branch, layout, Cancel, Undo/Redo and Save/reopen. Direct/grouped software probes supplement the native editor check. Runtime reuse remains R04; see the [acceptance checkpoint](r03t-task-continuity.md). |
| 4 | R03-V usable canvas space | Bounded viewport matrix verified: 3840×2160 at pixel ratio 1, 2560×1440 at 1.5, 1920×1080 at 2, plus compact 1280×720. A reproduced activity-shelf obstruction has a narrow fold-away control that retains history. Arranged stage actions and Save/export remain reachable with both side panels. Complete gate and Linux delivery results belong to the same paired PR checkpoints; physical OS scaling and arbitrary graph usability are not inferred. |
| 5 | R04 setup and repeated generation | Setup metadata repair delivered; bounded Linux cold/reuse/edit/adapter/recompute/release paths verified. Original constrained Windows, fresh GPU setup, different-artifact replacement and formal cached-adapter ownership remain held. See the [R04 acceptance checkpoint](r04-setup-repeat.md). |
| 6 | R05 original transport symptom | Historical matching Windows stacks identify Vite's WebSocket proxy; existing disconnect handling passes the bounded checks. The original reviewer's session remains unbound. See the [R05 trace](r05-refresh-transport.md). |
| 7 | R06 release and capability disposition | Open with explicit holds. Record each creative, access, runtime, hardware, provenance and publication limitation separately. Held cases are not passes. |

## R03-C first increment: actual developer acceptance

The initial native increment used unchanged backend `996208fe` and client
`e5c787eb`. An earlier live-browser receipt already established Add/enable,
execution, installed-source editing, Reload and refresh; its graph was assembled
through internal factories. Reuse that result at its original scope. The new
increment instead used ordinary Python-file upload, socket-driven discovery,
pointer wiring and library reopen through the production UI.

The small reviewed text processor declared a data-only runtime role, string
input/output and generated editable fields. It loaded no model, installed no
package and performed no network or filesystem action during execution.

The native journey established:

- Add of the actual Python file enabled its inspected source. Typed connection
  search found it from a string producer; dragging from its input also exposed
  compatible producers. The actual connection and inline prefix remained
  editable in the ordinary graph.
- Two ordinary Runs produced the initial and deliberately changed text. The
  developer edited only the newly installed source copy. Manage nodes showed
  changed source before explicit Reload authorized its new hash. Reload removed
  only that owned processor's cached result. Existing stale-approval regression
  tests retain their separate scope; no extra failed Run was needed.
- Save, page refresh and closing/reopening the named workflow preserved both
  node identities, positions, connection, authored controls, all 69 form fields
  and the executable graph. Genuine preview-result values changed as expected.
- Actual task/attempt/session/workflow bindings, approved source identities,
  cached text, durable output records, disk bytes and served text were joined.
  Generic consumed-field capture has no custom-field allowlist; that absence is
  retained rather than replaced with invented telemetry.
- Normal owned cache releases, disabling only the new QA extension and normal
  browser closure left the app idle with an empty cache and no downloads or
  scratch. All 393 earlier workflows, 92 output records, 87 preview slots,
  reusable Blocks and prior media were preserved. Only two genuine owned
  documents, two text outputs and their own ledger/queue changes were added.

Private selector and response-reader errors remain retained. No Run was
repeated to repair a reader. Two browser-harness-blocked frontend output-sync
posts are a limitation of that guarded journey; backend-created durable records
and served outputs were independently verified. This is bounded native
model-free developer evidence, not a comparative human usability verdict,
arbitrary extension safety guarantee, image/latent interchangeability proof or
hardware qualification. No product runtime change was needed for this loop.

The same two-node graph had usable ordinary input controls at 3840×2160,
2560×1440 and 1920×1080 effective CSS sizes with the existing sidebars/history.
This does not establish that every larger graph avoids an overlay. Treat the
canvas concern as conditional verification, not an assumed redesign.

## R03-D and R03-T next increments

R03-D's original browser regressions established that compact stages had no
visible direct editing action and that the library did not explain the two
authoring paths. The focused repair exposes **Edit stages** using the existing
separation/Undo transaction, labels collapsed reusable Blocks **Expand block**
using their existing expansion action, and calls reusable saves **Blocks**.
Library, Create Block, save and Python-import descriptions explain their purposes.
No graph format, executor, approval, node parameters or resource policy changes.
Focused browser checks preserve controls, source wires, Undo and Save/reload.
Visible and accessible action names agree. Delivery requires complete paired
gates, exact packaging and native verification at their recorded source heads;
the same PR descriptions retain those actual results and any remaining holds.

Trace the current common-control and stage-editing paths before changing them.
Keep task-level simplicity and real stage-level editing available together.
Use existing Block expansion, inspection, separation, generated fields and typed
ports; do not add another graph representation or executor. Test ordinary
pointer and keyboard access, Save/reopen, values, wires and Undo.

Keep these terms distinct: a reusable canvas Block packages authored nodes;
an executable Python extension supplies operator-authorized code. Contract-only
Blocks are not executable consent. File import copies source into the custom
folder: subsequent implementation changes must target that installed copy,
followed by explicit Reload. Discovery, refresh and workflow import never
authorize changed code. Source/dependency approvals remain hash-bound.

For model/task continuity, reuse declarations and existing authoring/model
evidence. Verify compatible prompt, fixed seed, actual image-source wires,
custom processors, unrelated branches and layout through review, Cancel,
Apply, Undo/Redo and a return path. Preserve genuine whole-pipeline exceptions,
guidance and conditioning requirements. A decoded-image socket is not proof of
latent or component compatibility. Any genuinely missing cached-model runtime
check requires fresh Linux-only serial admission and its own unchanged guard.

The R03-T editor check is now complete for the declared Klein pair. Actual
production-UI exports establish Cancel/Apply, a missing-image Run blocker,
ordinary insertion and binding of an existing image source, a return to text
and back, Undo/Redo, and Save/reload/library reopen. Compatible model identity,
prompt, fixed denoising seed, dimensions, steps, loading choices, Resize branch
and positions stay intact. A newly required image encoder retains its own
declared seed default; independent seeds are not silently coupled. Existing
direct/grouped planner checks supplement this bounded native modular case.
No product defect was established, no model ran and no package changed.
This does not establish live component reuse, arbitrary model/task or latent
compatibility, a new human verdict or public qualification. R04 retains the
separate runtime matrix; R03-V and R05–R06 remain at their stated scopes.

## R03-V: recover canvas space without clearing history

An ordinary pan can put a prompt field behind the fixed activity shelf even on
an unscaled 4K canvas. The original browser regression retained the actual
intercepting element at all four tested layout sizes. The narrow repair makes
the Session activity heading a keyboard-accessible disclosure. Folding it keeps
notifications and active-work status; opening it restores the same list. Explicit
toolbar download navigation opens it, while Clear keeps its separate meaning.
No sidebar redesign, graph format, saved values, node layout, execution, cache,
dependency or model-policy change was needed.

Production inspection also reproduced a selected-node toolbar covering the
activity heading after a viewport resize. Its existing placement now avoids
the measured activity rectangle, keeping both controls reachable as the list
folds, disappears or returns. Scheduled layout observations reposition only
the floating toolbar; selection and authored graph state stay unchanged.
The paired PR checkpoints distinguish the original held native journey from
the corrected software checks and subsequent delivery.

Checks cover ordinary pointer access to the previously covered field, Enter and
Space, all retained history, active work/failures/downloads, and unchanged graph
and viewport. The scaled Chromium checks use the corresponding CSS viewport and
pixel ratio; they are browser emulation, not physical monitor or operating-system
certification. Arranged multi-stage actions and Save/export were also checked
with both side panels at the four CSS sizes. This is bounded technical evidence,
not a new human verdict or a promise that every graph fits at a readable zoom.
Use normal pan, zoom and Arrange for a larger workflow. Existing R03-C/D/T and
accepted model/media evidence remain closed; no model run is needed for this
presentation repair. R04–R06 retain their separate acceptance requirements.

## R04: setup and repeat-generation acceptance remains required

The [R04 checkpoint](r04-setup-repeat.md) records the delivered metadata repair,
complete paired gates, actual CI scope and nine ordinary Linux UI runs. Reuse
those closed results. It distinguishes the first scope's idle time-bound closure
from the subsequent completed lifecycle scope, and retains missing ownership
telemetry and external acceptance as holds. Do not rerun closed models, gates or
deployment to change labels or repair private readers.

1. Review native uv/npm and guided setup against the executable contracts.
   Transformers and PEFT belong to initial setup. Browsing, discovery and Auto
   must not install packages. Test missing/damaged temporary metadata and clear
   repair guidance without modifying the operator's environment.
2. Distinguish installation, verification, activation/restart and current-worker
   readiness. Verify explicit allocator overrides, supported non-Triton eager
   routes and truthful unsupported optional-runtime states. Software mocks and
   an already configured machine do not establish fresh installation.
3. Map existing exact receipts to cold loading, unchanged result reuse,
   changed-seed inference, changed-prompt encoding and model/adapter replacement.
   Reuse closed boardwalk/car Auto evidence at its actual recorded source and
   full workload. Add inference only for a genuinely missing observation or a
   diagnosed changed execution path, using cached models and original inputs.
4. Preserve whole-workflow working demand, shared-storage/UMA accounting,
   capacity versus availability, actual component ownership, memory reserves,
   scratch/swap/time limits and normal scoped release. Missing telemetry remains
   absent. No workload reduction, OS tuning, forced retry or lower guard.
5. Keep the original 16GB GPU/32GB RAM Windows complaint externally held under
   the Linux-only model rule. Linux GPU success and Windows CPU CI cannot
   substitute for the user's actual constrained Windows generation acceptance.

## R05: original refresh/transport symptom remains separate

The [R05 trace](r05-refresh-transport.md) recovers ten exact matching error entries
from three archived Windows logs. Their prefixes identify Vite's WebSocket proxy,
and their stacks show Node socket writes. The earlier proxy lifecycle fix is
already present. A deliberately unfiltered control reproduces duplicate proxy
diagnostics; current focused proxy/refresh, connection, stale-response and genuine
upstream-failure tests pass. Historical production and Vite idle navigation records
remain bound to their original source rather than being relabelled current.

No new runtime defect or speculative transport change was established. The
reviewer's original October session still lacks a process/source/stack binding;
the independent historical match does not establish every possible cause of the
same error text. Preserve editing/task identity without duplicate dispatch. Use
the bounded capture guidance if it recurs; never suppress a broad error class or
repeat models merely to obtain a log. R06 remains separate.

## R06: release and capability disposition remains separate

- All 43 missing-model cases stay user-held. The added SSD is storage permission,
  not authority for model downloads, cache moves/configuration or mount changes.
- Boardwalk/car creative review remains separate from closed Auto technical
  proof. Accepted media and original rejected candidates remain preserved.
- Wan's fixed-deadline zero-output case and absent ACE melody codecs remain
  honest holds; no extended deadline, timbre substitute or unchanged retry.
- The retained Gallery 74-error report, recipe/prompt locks, old audio-duration
  mismatches, provenance, rights and public inventories need their own actual
  release checkpoint. Do not relabel private candidates as public qualification.
- CPU, NVIDIA, MPS and other advertised hardware require exact physical evidence.
  Unit CI and one large Linux GPU do not establish all-platform support.
- Final human review, template acceptance, publication and PR merging are
  separate decisions. Closed song/film/rocket verdicts are not reasked.

## Validation and operating boundaries

Author source only in the canonical Windows checkouts and keep changes in the
same two existing PR7s. Models run only on Linux and serially. No new agents,
chats, worktrees, installations, downloads, public uploads or paused campaigns.

For a diagnosed product change: retain a real failing regression, run focused
and related tests, then the complete required compatible client/backend gates.
Review exact bundle and semantic-ledger changes where affected, inspect actual
hosted jobs at exact heads, and use a new protected idle adoption if necessary.
Keep source/runtime/packages/configuration and unrelated data/cache owners
preserved. Never mutate an active model's source, graph or process.

Reuse completed gates for unchanged executable source. Plans, documentation and
private readers do not justify repeating closed model, deployment or full-gate
proofs. Maintain separate implementation, software, native, human and external
hold states for every row; do not report the whole feedback set as closed.
