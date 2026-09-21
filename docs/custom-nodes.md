# Developing custom nodes

In Developer, choose **Add from Hugging Face** or **Add local source** in the
**Workflows** dialog, or open **Nodes → Custom nodes** (also available in the Models
environment panel). Adding a source does not create or replace a workflow.
Choose a local Python folder, an HTTPS Git source, or a Hub Modular block.

For Hugging Face, enter the repository ID or its `https://huggingface.co/owner/repo`
URL, optionally with `/tree/<revision>`. Select **Resolve revision** to resolve a
branch, tag or commit (default `main`) to an exact lowercase 40-character commit.
Check the resolved source and commit, enter a module name, then stage it. Changing
the source or revision clears that result; cancellation discards late responses.
Resolution only reads Hub metadata. A bare repository ID with an exact commit can
also be staged directly. Git sources require an exact commit before staging.

**Stage source**
copies code and metadata into the backend's `custom/<Name>` directory with
execution disabled. It does not install Python dependencies or copy model weights.

Inspect the installed path, file hashes, code identity, declared dependencies,
inputs and outputs. Review the actual source files and license before selecting
the code-execution checkbox and **Enable code**. This imports trusted Python with
the backend's permissions. Custom web fields, if present, run in the browser
origin. This is not a Python sandbox or a source-code security audit.

Refresh, inspect, and cancelling a review do not import the submitted Python.
Developer workspace selection is not consent. An old directory manually placed in `custom/` also
requires review; its node identifiers remain the same after enabling. Existing
`custom/.disabled/<Name>` packages can be reviewed without deleting or relocating
their source. Names use a letter followed by letters, numbers or underscores.

## A small ordinary node

Stage `examples/custom_nodes/PromptTools` from this checkout using the module name
`PromptTools`. Its `main.py` declares a `NodeBase` subclass with `label`, `category`,
typed `params`, and an `execute` method. `__init__.py` exports that class. Inspect
and enable it, search for **Prompt Prefix**, and connect its string output to a
prompt input. The same node appears in the Nodes library in Creator and Developer.
No frontend code or separate executor is required.

Use `isInput: True` or `display: "input"` to expose an input socket. This
replaces its inline control; connect a value node such as **Text Value**. Outputs
use `display: "output"`. A type alone does not turn every configuration control
into a connectable input. Modular
sidecars' `input_names` and `model_input_names` declare those sockets directly.

Literal class metadata and module-level literal constants can be previewed
without imports. Dynamic metadata, `MODULE_MAP`, and `MODULE_PARSE` are resolved
only after approval. The executable classes must be available from `main.py`.
Import errors, including missing dependencies, remain visible in Custom nodes.

Edit the **installed source path** shown in the review panel. Staging copies a
folder; it does not create a link back to the original example or checkout.
Use **Review reload**, inspect the new hash, and **Enable and reload code**. Reload
invalidates this module's cached nodes and their transitive cached consumers,
preserving unrelated owners. A same-size quick edit and edits to relative helper
modules load fresh code. Changed code cannot run using its previous approval.
If you change field names or types, insert a fresh node and reconnect it as needed;
reload refreshes the registry but does not rewrite saved graph parameters.

Source resolution, inspection and listing remain available while a workflow runs.
Staging, enabling, disabling and reloading require an idle system: running,
queued or active metadata work returns a correction to
finish that work first. If an HTTP client disconnects after an approved import
starts, it cannot stop arbitrary Python safely; the execution lease remains held
until the operation finishes, including its separate metadata lease. Generic
field updates wait until the registry mutation finishes. Refresh sources to see the result. Failed imports
leave the module disabled with diagnostics. Python globals, native libraries,
threads and other import side effects may require a backend restart; disabling
or reloading cannot undo arbitrary code.

## Modular Diffusers blocks

Stage `examples/custom_nodes/ModularPrompt` to try a model-free upstream
`ModularPipelineBlocks` class. The same layout can be published on the Hub and
staged with its exact commit. Required files are:

- `modular_config.json` with `auto_map.ModularPipelineBlocks: "block.ClassName"`.
- `block.py` and any package-relative Python helpers.
- `mellon_pipeline_config.json` or `modiff_pipeline_config.json`, containing the
  `node_params.custom` typed UI, `input_names`, `model_input_names`, and
  `output_names` contract.

MoDiff validates and translates the existing Diffusers/Mellon metadata. The
custom-source adapter treats an omitted `model_input_names` as an empty list,
as Mellon does, while still rejecting invalid declared values. Normalization
does not rewrite the staged source or the exact bytes covered by approval.
After code approval, a block with pretrained component requirements and no declared
model inputs receives one **Models** socket. Connect **Pipeline Components** from
**Load Models**. Its tooltip lists the required component names. This interface
comes from the approved Python block, so it is not available in the import-free
sidecar preview. Existing sidecar fields and explicit model sockets are preserved;
insert a fresh node after upgrading if a saved instance lacks the new socket.

The approved entry point is imported in its own package, constructed with the native
Diffusers `from_config`, and executed through native `init_pipeline` and pipeline
calls inside the existing graph executor. It does not use upstream's shared
`diffusers_modules.local` alias, which can collide between local blocks/helpers
and retain stale bytecode. The upstream remote-code disable environment setting
still blocks activation.

Sidecar input/output names must match the actual imported block. Fields named
`out_<name>` map to upstream output `<name>`. Components must come from connected
MoDiff loaders and the existing ComponentsManager. This adapter does not silently
download or load a model from a block's default repository. Use the generic
Load Models node to select and load model components first. Framework
construction starts fresh pipeline state while connected weights remain shared.
All Modular Load Models routes publish their already-loaded component bundle.
Inactive optional components stay absent. Runtime validation checks actual component
types, including supported upstream Auto factories, rather than pipeline-family
names. A compatible class does not guarantee compatible tensor dimensions or tasks.

Stage `examples/custom_nodes/ModularImageReconstruction` for an executable example:
connect a decoded image to **Image** and the loader's **Pipeline Components**
to **Models**. Put it inside a Block and expose **Amount** as a control through
Configure Interface (0 to 1). Connect its Image output to Preview Image.
This block reuses an `AutoencoderKL` through its normal forward/offload hooks and
blends the reconstructed image with the source (0 retains the source, 1 uses the
reconstruction). It consumes no random generator and creates no model loader.
It honors the VAE's declared half-precision `force_upcast` setting for its forward
call and restores the original dtype even after an error.
Its component type is compatible with multiple image pipelines; VAEs of different
classes still require a suitable block implementation. Image dimensions must be
appropriate for the connected VAE's spatial scale.

When all pretrained component types resolve to installed official Diffusers or
Transformers classes, enabling the block also registers **Load Models — [block
name]**. This source-specific supplier uses the same component manager and node
executor as ordinary loaders; it does not add a model-family implementation.
It appears only after approval, alongside the block in Custom nodes.

Use it when the block needs additional weights, such as an annotator's model and
processor. Select a Hub repository, exact lowercase 40-character revision,
subfolder and optional weight variant for each component. Download those revisions
in Models before Run: this loader is cache-only and never installs packages or
executes model-repository Python. Select precision, device and offload policy, set
the workflow's Memory policy to **Custom**, and connect its **Pipeline Components**
output to the block's **Models** input. Existing connected model sockets continue
to work. Compatible loaded components are shared; source configuration changes
invalidate reuse without replacing another workflow's component.

Source approval is not resource qualification. Additional custom model suppliers
require Custom memory even when the processing block declares connected-component
resource use. Arbitrary Python component classes, local weight directories and
remote model code are not handled by this supplier; an approved block can still
accept compatible components from existing loaders. Model/type compatibility and
the upstream block's tensor/task semantics remain distinct.

The older Hub **User Node import** remains a declarative contract/library preview
and retains its fail-closed Dynamic Block checks. **Manage executable custom
nodes** opens this explicit extension flow. Importing a saved workflow or setting
its `trust_remote_code` field does not grant an extension approval.

## Dependencies and memory policy

`requirements.txt`, project dependencies in `pyproject.toml`, and requirements in
`modular_config.json` are shown with installed versions. Missing, incompatible or
unparseable declarations block enable. Direct URLs and pip command options require
manual review. Nothing installs packages automatically. Review imports too:
undeclared dependencies cannot be inferred completely from Python source.
Use the contributor runtime procedure for dependency changes; do not install into
or modify a sealed optional-runtime overlay. Reinspect after changing packages.

The code hash binds copied source files and declared installed dependency versions;
it is not a lock of every transitive package, an attestation of Python side
effects, or model qualification.

An optional `modiff_extension.json` declares one resource role:

| `runtimeRole` | Behavior |
| --- | --- |
| `data` | Author declares no model loading; the enabled node can run with Automatic memory. |
| `connected_components` | Author declares reuse of connected models; Automatic memory requires a connected reviewed model owner. |
| `manual` (default) | Resource use is unmanaged; select Custom memory in either workspace. |

Review this declaration with the code. It is an operator-approved extension
contract, not a measured memory guarantee. Automatic memory does not execute custom code
during inspection or grant new execution permissions. Arbitrary custom suppliers
of dimensions/model identities still need manual resource settings; they are not
promoted to the built-in preplanning evaluator. Automatic memory retains model owners in a
graph containing custom code rather than assuming that Python has released every
reference. Insufficient combined memory remains a blocker.

## HTTP flow

All code mutations use POST, are bounded, and retain the local single-user server
boundary. Local staging accepts a path on the backend machine. Source preview and
approval state are local administrative data; do not publish machine paths or
approval files in workflow packages.

For a Hub source, first `POST /custom_modules/resolve` with
`{"source":"https://huggingface.co/owner/repo","revision":"main"}`. The response's
`source` contains the normalized repository ID, `requestedRevision` and immutable
`revision`. Pass that exact identity to installation. This optional read-only
lookup does not stage files, import code or acquire the execution lease. Only Hub
model repositories are supported here, not Dataset/Space, file or subfolder URLs.

1. `POST /custom_modules/install` with
   `{"kind":"local","source":"examples/custom_nodes/PromptTools","name":"PromptTools"}`.
   Use `kind: "git"` or `"hub"` and `revision` for remote staging.
2. `POST /custom_modules/PromptTools/inspect` returns its current `module.codeHash`,
   files, dependencies and preview without importing Python.
3. `POST /custom_modules/PromptTools/enable` with
   `{"codeHash":"<exact inspected hash>","consent":true}`.
4. Use `custom.PromptTools.PromptPrefix` through the normal graph API. After source
   edits, inspect again and POST the new hash and consent to `/reload`.
5. `POST /custom_modules/PromptTools/disable` disables future execution and releases
   affected caches; files and model downloads are preserved.

`GET /custom_modules` and `POST /custom_modules/refresh` list sources without
reloading. Moving-branch `/update` returns an actionable rejection: stage a new
exact revision under a new name, review, then explicitly replace graph nodes.
The backend's `custom/.extensions.json` holds local approvals outside each source
package; a source cannot import its own approval by including that filename.
