# Workflow authoring and model selection

Creator opens Templates. Developer opens Workflows. Both use the same editable
nodes, Blocks, execution path and independent memory policy.

## Start a workflow

In Developer, search for an action or filter by Image, Audio, Video, 3D, or Text &
Utilities. Click a task once to create connected nodes. Select the model on the
loader and edit the prompt or parameters on the graph. Creating a graph does not
load models or install packages. Custom node source review remains available at
the bottom of the chooser and through Nodes → Custom nodes.

The backend resolves a task against published execution profiles. It prefers the
last explicitly selected compatible model when its exact artifacts and runtime
are available, otherwise an installed composable route, preferring the lower
estimated memory requirement among suitable routes. Automatic memory also
checks the existing read-only resource planner; Custom memory leaves that choice
to the operator. No selection loads weights or changes creative defaults. If no
route qualifies, the connected draft has an empty required model field. Select a
model there before running. Run independently rechecks inputs and resources.
Remembered selections are bounded local authoring preferences, not execution approval.

Composable routes expose meaningful independent nodes such as Load Models,
Encode Prompt, Denoise and Decode Latents. Tasks can need additional image, mask,
conditioning or audio operations. Whole-pipeline routes expose Generate/Edit
operations instead; those nodes do not claim independently replaceable denoising.
Qwen-Image 2.1 uses its supported whole-pipeline route with the compatible
installed Diffusers release. Its historical source review remains separately
pinned.

Curated fresh image recipes can request an exact backend `authoringRecipeId` on
the ordinary operation-starter endpoint. Its optional hash-covered `authoring`
record uses the existing role, edge and semantic form-binding tuples, plus
primitive `initialValues` triples. The backend binds the recipe to the reviewed
pipeline, task, execution profile and base specification. It declares retained
attention/VAE settings, the real Guider's CFG control and FLUX's separate embedded
scale. Qwen masked recipes explicitly select the owned whole-v1 compatibility
policy; outpainting declares the canvas utility and image/mask wires directly.
These are creation defaults, not a second executor or model-loading path. Clients
materialize ordinary nodes, seed untouched unconnected values and then apply
the authored form values. Existing authored or connected settings are preserved.
Requests without a recipe retain the ordinary developer starter defaults and
response envelope. Opening or running a saved graph never requests these defaults
or rewrites that graph. Unknown recipe IDs and mismatched pipeline/task/profile
selections fail closed.

Fresh Model Setup drafts expose the selected model's declared component weight
formats through the existing loader control. Native remains the default; models
with only Native keep the control hidden. This read-only schema projection
constructs no loader, changes no registered or saved node definition and grants
no runtime, fit or Auto qualification. Execution revalidates the selected format
and exact model artifacts through the ordinary loader.

Canonical example-library generation publishes ordinary executable leaves using
the existing projection for MoDiff's declared visual stage groups. Current public
control values, typed wires, operation ownership and stage roles are preserved;
user/custom Blocks are retained. This export mode validates the existing Block
hashes before projection and applies portable-data cleanup afterward. Ordinary
Save/export and the active canvas retain their grouped Block definitions. The
generator does not edit or rehash immutable nested definitions to sanitize them.
Pair-only regeneration also retains the published device fallback defaults on
unchanged, unambiguously matched executable nodes, rather than substituting the
generator host's defaults. Authored device values and fresh field schemas remain
intact; ambiguous matches or incompatible preserved defaults reject regeneration.

## Group nodes into a reusable Block

Select nodes and choose **Create block**. New definitions use Block V2 and can
contain existing Blocks such as Encode Inputs. Their nested contents, values and
crossing connections are retained. Existing saved definitions are not rewritten.
Creation errors appear inside the dialog. Public sockets and controls remain
available; the dialog lets you choose controls and rename ports.

Connection validation follows executable nodes inside containers. A loader and
denoiser grouped around an external encoder can exchange values without creating
an execution cycle. Required input checks also inspect these internal nodes; a
missing model connection blocks Run and identifies the affected node or Block.

When changing a model inside a Block, the change preview may retire untouched,
automatically derived outputs that the new route cannot provide and nobody uses.
Connected outputs, configured interfaces and preview bindings remain protected.
Incompatible contracts require repair before the change can be applied.

Scalar inputs declared with an editor (text, number, toggle or choices) retain
both their editor and optional socket. A connection supplies the execution value
and disables the literal editor; disconnecting restores the saved fallback.
The source is shown when available. Model and media inputs retain their sockets.

For a custom graph using Auto memory, **Graph ready · Auto check at Run** means
static graph checks passed. Run checks the exact backend resource plan before
submission. Use **Resources → Check Auto execution plan** to inspect it earlier.
Installed model files alone do not qualify a recipe; Custom memory retains your
chosen settings without claiming Auto qualification.

## Change the model

### Image Model Setup adapters

**Add LoRA** requires the selected image model's current backend capability,
execution profile, exact repository/revision pair and registered adapter sockets
to agree. Missing, refreshing or unsupported metadata disables the action with a
reason; opening and applying an adapter draft both recheck it. A LoRA socket alone
does not declare model support. The profile's optional `repository_revisions`
metadata contains bounded, unique pairs for its declared repositories; older
metadata remains readable but does not authorize new adapter edits.
An optional selected-profile `adapterAuthoring` policy can explicitly permit
editing adapters for its declared task, exact base-model pair and registered
adapter node. That permission is separate from aggregate model support and
runtime qualification. An optional `requiredControls` map constrains the
actual model owner before Add, Edit or Move. The current conditional Super
image route requires NONE/Auto Offload disabled/CPU weight staging, no
quantization or external components, and remote code disabled. Redux editing
requires effective NONE, no quantization/prequantized component, and declared
`device_map=none`. A declared pure execution recipe may override placement; a non-`none` raw
loader device map still wins over the recipe. Unbound `null` inputs qualify
only through the current registered optional-input schema. Missing,
malformed, unknown or ambiguous controls disable editing with a reason; no
field or execution recipe is inserted to obtain permission. Auto can use
only an already admitted candidate satisfying these controls. These checks
do not enable model execution or satisfy Cosmos safety/runtime readiness.
Without it, native aliases retain the canonical
pipeline's adapter restrictions. Malformed or mismatched policy disables edits.
Separate policies cover the dense pinned FLUX.1-dev, FLUX.1-schnell and
FLUX.1-Krea-dev native text-to-image profiles, FLUX.1-Kontext-dev editing, and
FLUX.2-klein-4B text-to-image and editing. Each permits only its own reviewed
repository/revision pair; sharing a pipeline class does not transfer permission.
Multi-reference authoring uses the current backend owner's explicitly declared
edit execution mode, with the actual task and profile still checked. Saved graph
hints cannot grant that mapping. Compatible FP8/NVFP4 models, Klein Base/KV and
unrelated tasks receive no permission from these policies. Existing runtime
routes and qualification remain unchanged. Add also checks the owned-row limit. Reaching that limit
leaves permitted existing rows editable and reorderable.

Existing unsupported adapters stay visible with their values and connections.
Their editors and ordering controls remain disabled; **Remove** can repair the
graph through the ordinary Undo transaction, subject to shared supplier and
sealed interface checks. Opening a saved graph never discards adapters or regroups
its stages. Reviewed whole-image adapters preserve the owner-to-generation scope
through their actual pipeline sockets rather than becoming operation stages.

Fresh native image recipes group their declared image decoder and connected
Preview into **Image Output**, including decoders whose backend operation uses
the generic `diffusion.decode_latents` identity. This changes presentation only;
the executable decoder, output fields and Preview edge remain intact.

### Guidance control visibility

Removing a Guidance row through Configure interface hides the row, not the
underlying operation or its effect. The current value is retained on its original
stage. Explicit removals are recorded by `{nodeId, fieldId}` in the optional
instance `presentation.removedControlBindings` list and survive reload and dynamic
field refresh. Re-exposing a control clears its removal record. Reusable-node
definitions can carry the same optional `removedControlBindings` list as initial
presentation; it is excluded from execution identity. Both client and backend
validate bounded, unique bindings. Historical omissions without a removal record
are not guessed to be deliberate removals, and opening a workflow does not rewrite it.

### Model selection

The model field of a canonical loader opens **Choose model for Load Models**.
Search by model or repository and optionally filter to Downloaded. Names precede
repository IDs; download status is a trailing badge. Selecting a row resolves the
backend starter and updates the owned operation graph through the normal graph
transaction. Edited compatible values and unrelated branches remain intact.
An untouched starter can change between composable and whole-pipeline routes
without leaving disabled old nodes: its internal edges and controls are checked
against a backend-resolved baseline, and its unique media preview is reconnected.
Edited values, custom wiring or ambiguous outputs use the preservation/review
planner. Schema or connection differences requiring review are shown before applying.
Undo/Redo and saved workflow reload use the same graph history and persistence.

Legacy managed Task changes apply the existing fallback for an unavailable
default/CUDA device using the current backend runtime metadata before building
the graph. CPU offload settings normalize through the ordinary form policy;
authored prompts remain intact. Explicit unavailable XPU, MPS or unknown device
choices remain visible for validation. A newer document or creative edit still
cancels a pending graph operation.

A raw implementation/component loader has a picker restricted to compatible
installed artifacts. In a connected workflow, model selection derives the Pipeline
Type and applies both atomically. A raw implementation loader exposes Pipeline Type
only when no workflow/model-choice contract can infer it. Clicking a row
applies its repository to that field. **Manage model files** opens management;
copying a repository ID is not required for selection. Canonical switching does
not broaden a legacy loader's accepted runtime contract.

Model choices remain backend-declared. Standard and Modular routes for one
repository share a model row. The current implementation stays primary; otherwise
ready composable operations are preferred. Other implementations remain available
under the row's disclosure. An explicit successful selection is remembered per task.

**Restore model defaults** resets unconnected creative controls using the selected
profile's starter. Model ownership, memory settings, media and custom branches
remain intact; Undo restores the prior graph. Public Block model controls resolve
the owning internal loader before using the existing atomic Block transaction. Imported custom sources still require explicit review
and approval; opening a picker or choosing Developer does not enable code.

## Names, examples and library

Raw loaders use distinct names: **Load Image Pipeline**, **Load Audio Pipeline**,
**Load Modular Components**, and **Load Model Component**. Canonical workflows use
**Load Models**. Other corrected names include **Prepare Outpaint Canvas**,
**Generate Image with Control**, **Preview Latents**, and **Export Video Asset**.
Names are selected by execution identity, not guessed from class-name substrings.
Serialized identities and custom titles are preserved; previous built-in labels
remain searchable.

New resolved operation schemas receive task examples. FLUX Schnell, FLUX.2 Klein
4B, Z-Image Turbo, original Qwen Image, Stable Audio Open and AudioLDM2 have
concise creator-derived examples with source links. Earlier Qwen 2.1 examples remain intact. Other
mapped tasks use labeled MoDiff examples. Numeric settings continue to come from
reviewed execution profiles. This does not migrate saved prompts or replace edited
values. Full creator-example coverage and all legacy insertion paths remain tracked
work; the presence of a sample does not qualify a model's output.

The sidebar offers **Start**, **My workflows**, **Example workflows** and
**Recovery drafts**. Start uses the same task browser as the Developer launcher;
it creates a separate tab without replacing the current graph. Open workflow
shows My workflows. Lists remain bounded and paginated.

New documents autosave as recovery drafts. Explicit Save promotes them to My
workflows; a delayed autosave cannot demote a saved document. Existing documents
without an intent marker are treated as saved without rewriting their files.
Unknown historical documents are not reclassified or deleted.

Shared select, combobox and multiselect menus use an 18rem preferred cap, further
limited by available viewport space. The cap is passed through Headless UI's
anchoring constraint so its inline styles cannot override the intended limit.

## Verification boundary

Regression tests cover naming, task creation, model changes, edited prompts,
Undo/Redo, reload, cancellation and long option lists. Backend tests resolve
contracts without loading weights. These checks establish authoring behavior;
they are not all-model image/audio output qualification. Video output qualification
and Windows Qwen on 16GB VRAM / 32GB RAM remain separate pending acceptance.

Native Modular execution has also been checked with Z-Image Turbo and FLUX.2
Klein 4B using the same generic graph. An unchanged Z-Image rerun reused all
nodes; changing its seed retained models and prompt encoding; changing its prompt
retained models. Switching to FLUX preserved the edited prompt and generated a
new image. This is representative execution evidence, not every-model coverage.
A first Run click immediately after switching and editing parameters failed to
submit during this check; a later click succeeded. That timing issue remains
under investigation; a fresh-session first-click repetition passed.

A later instrumented repetition of baseline, unchanged, seed, prompt and FLUX
model-switch runs submitted successfully on the first click in all five cases.
The original intermittent failure is retained as unresolved; this is diagnostic
evidence, not a claimed fix.

## Final representative authoring acceptance

The final production-build journey created a Z-Image Turbo text-to-image workflow
from the task browser, restored its creator-derived prompt and completed all five
independent Modular nodes on the first Run click. The workflow was explicitly
saved, a second workflow was created from sidebar Start, and the first workflow
was converted to a user Block. Its public model control then switched the same
Block to FLUX.2 Klein 4B and completed all five nodes on the first Run click. Both
1024×1024 outputs were visually reviewed: the restored Z-Image prompt produced
the intended portrait and the FLUX prompt produced a cat with the legible requested
sign.

That journey exposed one compatibility defect before it passed. A derived Block
interface can contain optional fields from its source model that the target model
does not implement. Model replacement now removes only unavailable fields that
were generated automatically, remain at their defaults, have no stored boundary
value and have no outside connection. Explicitly configured interfaces, connected
inputs, sealed controls and edited values continue to reject an unsafe switch and
use the existing review path. The focused operation-authoring suite covers both
the allowed and protected cases; the final client check, production build and
bundle limits passed. The mirrored backend web bundle matches the tested client
distribution byte-for-byte.

This is representative acceptance of task-first creation, defaults, save/draft
separation, Block ownership and cross-family Modular switching. It does not expand
the all-model image/audio qualification boundary above. The browser harness ended
with a bookkeeping assertion that expected the still-open recovery draft in the
mocked saved-document list; application execution, output capture and error checks
had already passed, and only the explicitly saved workflow had correctly been
written there.


## Private ancillary execution selections

An optional backend-authored ancillary operation selects a subset of the current
ordinary graph. Its exact parent authoring recipe and base execution spec have
separate content hashes. Explicit Preview targets the declared real leaf and its
actual dependencies; ordinary Run excludes only its declared Preview-only roles.
Stale or malformed claimed scopes refuse before preparation. Historical graphs
without this optional declaration retain their existing behavior.

These private contracts do not publish a model/profile, grant Auto admission,
authorize owner release or issue a proposal Apply receipt. Parent media sources
retain their actual order; successful decoding and normalized consumed media must
still be verified separately. Purpose metadata never authorizes global cleanup.
