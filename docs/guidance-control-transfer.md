# Guidance controls during model changes

The operation authoring planner preserves a changed guidance value only when the
backend declares its meaning and the prospective destination consumes it under a
compatible policy. This is authoring metadata, not a replacement guider, an
execution permission, or tensor compatibility. Existing saved workflows and
creator recipes retain their execution values.

## What the declarations mean

Version 3 operation ports can include an additive `semantics.control` declaration.
Older version 3 contracts without it still parse. The declaration names the
technique, parameter, compatibility scope, scale meaning, enabled policy,
formulation, selector and negative-conditioning policy. References point to real
input fields in the same operation contract; the client rejects unknown keys,
invalid policies, missing referenced inputs and incompatible input types.

The reviewed adapter projection lives in
[guidance_operation_semantics.py](../modiff/guidance_operation_semantics.py).
It describes exact executors and their pinned upstream behavior:

- Classifier-free guidance mixes conditional and unconditional predictions.
  Original and Diffusers formulations remain distinct. An actual native guider's
  selected technique, enabled switch and formulation determine the policy.
- Qwen's normalized classifier-free prediction has its own compatibility scope.
  A similarly named plain CFG field does not authorize the same transfer. The
  direct Layered adapter's explicit normalization selector is inspected before
  its reviewed plain-CFG variant can transfer.
- Embedded distilled guidance is a model input, separate from true CFG. Its
  consumption can depend on the loaded transformer's configuration. Discovery
  alone cannot establish whether a FLUX model consumes that scale.
- Standard image adapters can rename controls. Reviewed declarations describe
  the consumed meaning rather than relying on `guidance_scale` versus
  `true_cfg_scale`. A secondary-scale override must actually be selected.
- A standard pipeline's negative prompt is a supplied condition even when its
  string is empty. Missing, connected or opaque conditioning stays unresolved
  during a scalar transfer. Conditioning tensors retain their own pipeline and
  loader ownership; transferring a scalar never transfers those tensors.

Unsupported techniques and unreviewed controls remain undeclared. An operation
name or field label does not fill that gap. This metadata does not change
positive/negative encoding, guidance math, model loading, or default values.
Standard SD/SDXL classes also require inspection of the loaded UNet configuration:
`time_cond_proj_dim` can replace CFG with a time-conditioned guidance embedding.
A class name and scale threshold alone do not authorize that transfer. Those
class-level controls remain unresolved until a selected-model predicate exists.

## What a model-change preview preserves

The planner checks both the source policy and the destination policy **after the
proposed value transfer**. For example, copying a native enabled guider's scale
of zero into a pipeline that enables CFG only above one changes whether guidance
is used. The destination keeps its default and the old value remains available
for review instead of being copied automatically.

Compatible declared scalar meanings can transfer across families. Different
formulations, normalized versus plain CFG, distilled versus true CFG, unresolved
selectors or connected policy inputs produce a retained-value finding. The
preview reports affected values and connections before the existing atomic graph
transaction commits. External value suppliers remain in the graph if a wire
cannot safely reconnect. Undo restores the earlier graph.

Keeping the exact model, executor, binding and control declaration preserves its
existing unresolved runtime policy and wires. Adding an input or regrouping a
stage must not discard an edited distilled scale merely because authoring cannot
read the loaded model's configuration. This identity-preservation rule does not
permit transfer to a different repository, revision, profile or executor. An
exact archived route can likewise recover its saved settings when selected again.

Returning to an exact archived model and task also restores its authored hidden
runtime values as one coherent set. The planner checks the owner identity,
executor, binding, complete port declarations, accepted values and absence of
incoming wires before restoring them. If any setting cannot pass those checks,
the archive remains available and the preview explains that the selected route's
defaults apply. A different model, task or explicit defaults reset does not
inherit those hidden settings. This preserves paired runtime modes without
moving component handles or authorizing a new execution route.

Old snapshots without control metadata retain their existing execution. A model
switch between different pipeline classes does not infer guidance equivalence
from legacy labels; unavailable meanings require review. No package installation
or model load happens during planning.

## Verification boundaries

The backend tests generate reviewed native and standard contracts from the actual
operation catalog. The shared JSON fixture is parsed by the client; authoring
regressions cover safe aliases, enabled/formulation mismatches, empty negative
text, unresolved connected policies, external wires and unchanged model identity.
See [backend tests](../tests/test_guidance_operation_semantics.py) and the client
[authoring checks](https://github.com/sdevil7th/MoDiff-client/blob/fix-ui-ux-issues/scripts/operation-authoring.test.mjs).

These checks establish contract and graph behavior. They do not qualify new
models, prove image equality, or establish accelerator fit. Those outcomes follow
[image-template validation](image-template-validation.md) and
[three-machine application validation](three-machine-app-validation.md).
