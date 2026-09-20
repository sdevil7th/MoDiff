# Service prototyping with an API graph

In the Developer workspace, choose **Export → Service package**. MoDiff lowers the same graph
used by API graph export, including expanded Blocks and Modular composition.
Name the scalar controls callers should supply and the preview outputs they
should receive. Leave a field blank to keep it internal. Named inputs have no
exported default: supply every value on every invocation. At least one preview
output is required. The Creator workspace keeps its compact Export menu.

The resulting `modiff-service-v1` file contains the existing API graph, a named
interface, and an observed execution manifest. It runs through the local MoDiff
server's normal graph queue, caching, resource planning, cancellation and output
persistence. It is **not standalone Diffusers Python**, a hosted endpoint, or an
authenticated multi-user server. Arbitrary-graph Python generation remains a
separate feature. Keep the server at its supported loopback boundary.

## Model-free example

Start the backend with the [developer setup commands](developer-setup.md). In a
second terminal, from the backend checkout on Linux:

```bash
.venv/bin/python -m modiff.service inspect examples/service/text-api.json
.venv/bin/python -m modiff.service build examples/service/text-api.json --interface examples/service/interface.json --output text-service.json
.venv/bin/python -m modiff.service run text-service.json --inputs examples/service/inputs.json
```

On Windows, replace `.venv/bin/python` with `.venv/Scripts/python.exe`. The client
uses standard-library HTTP and never imports the node registry or model runtime.
`--output` creates a new JSON file and refuses to overwrite one. The example uses
Text Value → Data Viewer, needs no models, and returns a named `text` preview.
No service file generated from a local runtime belongs in the repository.

The interface file maps names to exact lowered node/field identities:

```json
{
  "inputs": {"prompt": [{"nodeId": "prompt", "field": "text"}]},
  "outputs": {"text": [{"nodeId": "preview", "field": "preview"}]}
}
```

A CLI-authored binding can target multiple fields of the same declared type,
useful when a graph shares prompt or seed values. The dialog assigns one target
per name. Inputs cannot replace connected fields or model/code identity controls.
Outputs select persisted `ui_text`, `ui_image`, `ui_audio` or `ui_video` fields;
add a preview node for tensors or other values. Output values are arrays of
records so repeated loop outputs retain their existing representation.

## HTTP contract

`POST /service_package` accepts three bounded JSON operations:

- `{"operation":"inspect","graph":...}` returns `inputs` and `outputs` candidates.
- `{"operation":"build","graph":...,"interface":...}` returns `package`.
- `{"operation":"prepare","package":...,"values":{"prompt":"..."},"sid":"service_example"}`
  returns `graph`, ready for the existing `POST /graph` route.

All responses have `error:false` on success; invalid contracts return HTTP 400
with an actionable `message`. The body limit is 8 MiB; duplicate JSON keys and
non-finite numbers are rejected. Inspection/build/prepare do not execute a graph,
install code, enable extensions, or download weights.

Post the returned graph unchanged to `/graph`. Its `task_id` identifies the run.
Poll `/queue`, then fetch `/runs/{task_id}` on completion. The CLI returns only
matching task/node/preview values and durable URLs, without copying private graph
snapshots into its response. A timeout prints the task ID and leaves execution
alone; inspect that run before retrying to avoid duplicate work. Existing stop
and queue controls apply. Media URLs refer to this local server and are not
permanent public hosting links.

Requirements are checked during preparation **and when execution begins**. A
queued package cannot silently pick up different approved custom source or
installed dependencies. Named values change no graph topology. Auto gets a fresh
graph-bound plan receipt for those values; the existing executor resolves current
hardware/resources and refuses unsupported plans. Manual retains exported
execution settings. Neither mode changes model or creative inputs to fit memory.

## Portability and limits

The manifest records backend commit/source fingerprint, Python minor version,
managed accelerator profile and dependency-contract digest, observed installed
package versions, required optional-runtime profile IDs, explicit Hub model
references and immutable revisions, and enabled custom-node code/dependency
identities. Cataloged repositories reuse the reviewed pin. Unknown repositories
require an explicit 40-character revision. Model selectors connected to other
nodes require a literal pinned selection before this version can be exported.

This is a strict observed-environment snapshot, not an automatic environment
installer or a claim of bit-identical images across devices. Recreate the same
reviewed profile/packages and backend source before replay. Extra or changed
installed packages currently require re-export. Cross-profile execution requires
reviewing and exporting a package on the target profile. The service manifest
never authorizes code: stage/review/enable custom nodes independently on each
machine. It does not bundle custom source, weights or their licenses. Custom node
authors must expose external model dependencies explicitly; hidden downloads in
arbitrary Python cannot be inferred by graph inspection.

Session IDs and UI snapshots are omitted. Execution parameters and execution
hints are retained; the exporter refuses detected credentials, local model
selections and local paths in portable material. Required inputs let callers
provide file identifiers or private text at invocation without embedding them in
the package. Opaque/custom values and file upload convenience controls are not
part of the first scalar interface. Review remaining prompts and metadata before
sharing: arbitrary secrets written as ordinary prose cannot be identified
reliably. Export errors are not permission to remove execution constraints;
resolve the named field and export again.

The committed example and HTTP smoke prove a model-free API round trip. Diffusion
model output, optional-runtime availability, Windows execution and particular
GPU memory envelopes require their own validation evidence.
