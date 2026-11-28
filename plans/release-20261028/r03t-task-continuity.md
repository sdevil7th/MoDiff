# R03-T: same-model task continuity

Editor acceptance checkpoint, 7 November 2026. This supplements the original
human-accepted R03 prototyping journey and the separate R03-C/D work. It does
not reopen those accepted or closed results.

## Verified behavior

The declared FLUX.2 Klein 4B modular text-to-image and image-conditioned tasks
use the same immutable model revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`. Backend-generated task contracts,
the existing graph-change planner and ordinary production-UI gestures were
used; no alternate executor, graph submission or hidden-store mutation was
needed.

The production editor check established:

- Review followed by Cancel preserves the executable graph, all 69 authored
  form fields and node positions. Apply changes only the declared task route
  and required stages while preserving compatible authored values.
- The image-conditioned task blocks Run while its required image is missing.
  Ordinary typed-socket insertion of Load Image and binding an existing retained
  image enables Run. Run was inspected, not pressed.
- Model identity and revision, prompt, fixed denoising seed, dimensions, steps,
  guidance, loading choices and the user-added output Resize branch survive the
  round trip. The source node, actual image wire and positions return with the
  image-conditioned route.
- Undo and Redo restore their respective executable graphs. Explicit Save,
  page reload and closing/reopening the saved workflow preserve the graph,
  source binding and authored form values.
- Backend-generated direct and grouped planner probes also preserve compatible
  authored values, processing and unrelated branches. These are software checks;
  they do not add native inference or arbitrary custom-node compatibility.

The native production check ran at backend `04e485f6` and served client
`21ef1325`. The served 119-asset bundle is byte-identical to canonical client
`565e2d74`; its later changes affect only CI, mocked tests and documentation.
Seventeen genuine workflow exports, their ordinary requests and a separate
read-only audit bind the required editor behavior. Normal QA closure preserved
all earlier saved work, outputs, media, custom source and approvals, runtime,
packages and configuration. Only genuine owned QA workflow additions occurred.

No product defect was established, so no product repair, deployment, model run,
download or installation was performed for R03-T. Existing complete paired
gates are reused at their actual unchanged executable source; their raw source
bindings are retained.

## Seed and task semantics

The fixed Denoise seed remained `8307`. The newly required ImageEncode stage
correctly used its independent declared default `0`: this route passes image
latents, not a shared generator state. An initial private capture incorrectly
required every seed to equal the Denoise seed. Its failed receipt is preserved.
Correcting that reader required no recipe change or model retry. Only unfinished
editor steps were continued from the genuine applied-task export.

The declared task is image-conditioned editing. Its existence does not imply
that every model offers the same controls, that every edit task is ordinary
img2img, or that decoded-image sockets establish latent compatibility. Supported
whole-pipeline routes and their required controls remain explicit.

## Manual editor check

Use a workflow with a cached model that declares both tasks; this checkpoint
uses Klein's modular profile. These steps require no model Run or download:

1. Set a distinctive prompt and fixed Denoise seed. Add an output Resize branch
   if desired, then Save.
2. Open **Change model / task** on Model Setup. Keep the compatible model
   selection, choose the declared image task and preview the change. Cancel
   once and check that the authored graph remains intact.
3. Preview again and Apply. Check the required-image guidance, connect Load
   Image to the image input and select an existing image. Check Run readiness.
4. Preview and apply the text task, then use Undo/Redo. Return to the image task
   and confirm that the source connection and compatible values are restored.
5. Save, refresh and reopen the saved workflow from My workflows. Check its
   values, source connection and processing branch.

These are authoring checks. Actually running a model requires ordinary runtime,
trust, input and resource admission.

## Remaining acceptance

R04 owns cold loading, unchanged reuse, changed-seed inference, changed-prompt
encoding and model/adapter replacement. Existing native outputs are reused at
their genuine recorded scope; this editor check does not prove live component
reuse or qualify additional hardware. Conditional R03-V, R05 transport attribution
and R06 release/capability disposition remain separate in the
[saved work plan](mellon-follow-up.md).

All 43 missing-model cases remain user-held. Accepted media and human verdicts
remain unchanged. Private source material and raw receipts remain outside Git;
this checkpoint grants no new creative, template, publication or merge approval.
