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
Qwen-Image 2.1 uses its supported whole-pipeline route at the reviewed Diffusers pin.

## Change the model

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

A raw implementation/component loader has a picker restricted to compatible
installed artifacts. Select its Model Type first when required. Clicking a row
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
