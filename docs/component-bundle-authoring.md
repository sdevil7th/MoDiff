# Optional component bundle wiring

On a reviewed Qwen-Image text-to-image graph, open the model loader's inspector
and choose **Workflow stage actions → Use component bundle**. The action can
replace its individual component fanout to Encode Prompt, Denoise and Decode
Latents with the existing `pipeline_components` output. **Expose components**
restores individual sockets. Each edit uses the ordinary workflow transaction,
supports Undo, and retains expandable visual groups and their configured controls.

This is an opt-in prototype for `QwenImageModularPipeline` / `text_to_image`.
Starter and Gallery defaults keep their original individual wires. No model,
loader, pipeline, component owner or graph executor is created by the action.

## The actual reduction

The reviewed graph has four loader-to-stage component edges: text encoders,
transformer, scheduler and VAE. The compact form uses three bundle edges, one
per consuming stage. That removes one wire. Guidance and conditioning connections
remain real and unchanged. Model Setup and Image Output groups can separately
reduce visible canvas objects; those groups do not reduce executable stages.

The public operation contracts declare exact named component members and their
types. A required individual input remains required and can declare a
`semantics.suppliedBy` alternative. Client parsing checks the referenced bundle
input and exact raw member coverage/types. Required-input readiness and Graph Fix
use the same enabled, scoped, typed supplier declaration rather than reporting
missing individual wires after a valid bundle connection.

The ordinary stage wrappers project only their existing roles:

| Stage | Bundle members | Existing individual input |
| --- | --- | --- |
| Encode Prompt | `text_encoder`, `tokenizer` | `text_encoders` |
| Denoise | `transformer`, `scheduler` | `unet`, `scheduler` |
| Decode Latents | `vae` | `vae` |

The shared Guider remains separately supplied. It is not inserted into the text
encoder role simply because the upstream encoding block also consumes it.

## Preserve ownership and explicit edits

The authoring action reduces only exact fanout from the original declared loader.
It skips a consumer with a competing component supplier, multiple drivers, or a
literal component override whose runtime identity cannot be proved by authoring.
Those ordinary connections stay intact. An alternate bundle supplier also stays
intact. Breakout restores only unsupplied raw inputs and keeps explicit overrides
and unrelated branches.

At execution, the backend validates the existing sealed bundle, demanded managed
IDs, component roles and loader generation before installing components or
reusing cached results. Temporary projections carry the same loader token.
Explicit runtime overrides must have that same token, role and exact managed
members; independently loaded replacements use the ordinary breakout path.
Missing, evicted, forged, mutated or mixed-owner handles fail before native
initialization. No ambient or most-recent loader can supply a missing member.

The cache compares original input/bundle identity before validating current
residency and route ownership. Fresh projection dictionaries do not become a new
cache identity. Existing unbundled execution follows its previous path. Ordinary
Auto owner/task detection continues to inspect the same graph; this feature does
not qualify a new model or resource profile.

## What is deliberately scoped

The prototype does not substitute bundles for image-conditioned, masked, control
or reference routes. Those paths require their selected task's actual component
and route-state contracts. Registered whole-workflow stages already have their
own bundle boundary; this action does not retrofit or reinterpret it.

Backend [boundary tests](../tests/test_qwen_component_bundle.py) cover actual stage
installation, exact managed objects, same-token route issuance, repeat-cache
validation, eviction and mixed-owner rejection. Shared client/backend fixtures
and the client authoring/readiness/browser checks cover graph behavior separately.
They do not establish full-model pixel parity or accelerator fit. Use
[image-template validation](image-template-validation.md) for output comparisons
and [three-machine validation](three-machine-app-validation.md) for platform tests.
