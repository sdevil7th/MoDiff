# Workflow authoring and model selection

Creator opens Templates. Developer opens Workflows. Both use the same editable
nodes, Blocks, execution path and independent memory policy.

## Start a workflow

In Developer, search for an action or filter by Image, Audio, Video, 3D, or Text &
Utilities. Click a task once to create connected nodes. Select the model on the
loader and edit the prompt or parameters on the graph. Creating a graph does not
load models or install packages. Custom node source review remains available at
the bottom of the chooser and through Nodes → Custom nodes.

The initial choice is deterministic among backend-declared routes: ready runtime,
complete installed artifacts, an execution adapter, then composable operations.
Ties use the model label and stable choice ID. This is an authoring default, not
a hardware qualification or a memory-fit claim. Run still validates the model,
inputs, dependencies and resource recipe. Remembered choices and unbound task
drafts are not implemented in this correction.

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

Model choices remain backend-declared. Standard and Modular variants may currently
appear as separate rows. Imported custom sources still require explicit review
and approval; opening a picker or choosing Developer does not enable code.

## Names, examples and library

Raw loaders use distinct names: **Load Image Pipeline**, **Load Audio Pipeline**,
**Load Modular Components**, and **Load Model Component**. Canonical workflows use
**Load Models**. Other corrected names include **Prepare Outpaint Canvas**,
**Generate Image with Control**, **Preview Latents**, and **Export Video Asset**.
Names are selected by execution identity, not guessed from class-name substrings.
Serialized identities and custom titles are preserved; previous built-in labels
remain searchable.

New resolved operation schemas receive task examples. Qwen 2.1, FLUX Schnell and
Z-Image Turbo have concise creator-derived examples with source links. Other
mapped tasks use labeled MoDiff examples. Numeric settings continue to come from
reviewed execution profiles. This does not migrate saved prompts or replace edited
values. Full creator-example coverage and all legacy insertion paths remain tracked
work; the presence of a sample does not qualify a model's output.

The sidebar separates **Example workflows** from **My workflows**. Saved documents
stay searchable and paginated in their own bounded list. No existing documents
are deleted. A saved/draft/recovery migration is separate pending work.

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
