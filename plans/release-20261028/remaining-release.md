# Broader remaining work and release boundaries

These tasks remain visible after the [28 October candidate plan](README.md).
They are not silently dropped from the originally requested whole-app release.
Do not start them in the Windows manual-testing window. A later continuation
needs an explicit finite selection and updated source/evidence inventory.

The retained 27 October catalog census describes 80 public templates, 203 ordinary
workflows, 149 standalone/nonpublic-source rows, 58 resource tuples and seven
hardware profiles. Fifty-four ordinary workflows overlap public source templates;
these are not 283 independent model jobs. The 149 standalone rows include 100 image,
33 video, six audio and ten JSON outputs, with 20 deterministic builtins across
that set. It spans 91 model types. Refresh this census from the final source before
execution; the historical source pair is not current qualification.

## B1 — complete the canonical execution-evidence path

**Implementation incomplete. Budget: 4–8 hours for one bounded, genuinely native
tranche; approximately 1–3 additional working days for wider task/input/builtin
coverage, subject to the first tranche's findings.** This is tooling effort before
the missing model campaign, not an estimate for all 149 outputs.

**Where:** BE `modiff/execution_input_provenance.py`, `NodeBase.py`,
`modules/DiffusersRuntime/main.py`, `modules/DiffusersImage/call_inputs.py`, affected
loaders/output producers and `tests/test_canonical_field_accounting.py`; FE
`scripts/live-proof-provenance.mjs`, `release-contract-core.mjs`,
`release-contract.mjs`, `template-qualification.mjs`, their fixture/negative tests
and `src/studio/resolvedExecutionInputs.ts`.

**Current limitations:** input-free initial accounting is deliberately narrow;
input-backed canonical qualification and deterministic builtins are explicitly
rejected. A strict five-node DDIM accounting fixture is not universal graph support.
The actual three-node UI smoke did not supply the unchanged canonical graph's
complete evidence. Loader execution-profile/recipe accounting and generator
presentation-contract identity remained incomplete. Actual 32×32 bytes and legacy
1024×1024 form metadata must remain distinguishable.

**Approach:**

1. Reproduce the smallest real producer → normal export → reader failure without
   editing original packages. Trace which values were requested, normalized,
   consumed, inactive, superseded or merely presentation metadata.
2. Fix the actual producer/normalizer/join only where causality is established.
   Preserve bounded payloads, privacy, legacy normalization and explicit unknowns.
3. Prove one unchanged canonical graph through the normal UI and original exports.
   Confirm exact graph/default/prompt/model/component/task/output identities and
   actual measurements. Do not infer missing fields from the current catalog.
4. Add input-backed evidence at execution time, binding the consumed input bytes;
   a later file copy is insufficient. Add builtin semantics where model/accelerator
   identity is genuinely not applicable, with real execution and output proof.
5. Extend accounting by declared task policy, including multiple outputs/owners,
   only with positive and negative cases. Keep retry/fallback, quantization and
   unsupported joins explicitly held until intentionally covered.
6. Reuse the existing executor and recorder. No second workflow representation,
   fake model identity, loosened hash validator or automatic qualification promotion.

**Exit:** native receipts pass the maintained reader for each newly supported scope;
unknown IDs, changed graphs/inputs/model sets, truncated evidence and unsupported
claims fail. Quality, hardware fit, rights and publication still require separate
acceptance. A recorder pass does not approve a template.

## B2 — reconcile the historical Gallery before scheduling reruns

**Mixed evidence/recipe/decoder gaps. Budget: 3–6 hours for the recoverable diagnostic
subset; 1–3 additional working days may be needed for actual replacements.** Media
generation/access and human quality approval are additional dependencies.

**Where:** FE `scripts/template-gallery-harness.mjs`,
`template-gallery-repair-evidence.mjs`, `template-gallery-repair-contract.mjs`,
`live-proof-provenance.mjs`, `src/studio/templateExactness.ts`, `templates.ts`,
`templateOperationSelections.ts`, asset-source/rights descriptors and corresponding
tests. Preserve the original public Dataset and original approved bytes.

The latest retained report had **83 overlapping errors across 70 examples**.
All 356 approved local files were recovered; the old missing-byte problem is not
the current blocker. This is neither 83 mandatory reruns nor 49 accepted repairs:

| Group                     | Current issue                                         | Correct next action                                                                                                |
| ------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 49 historical candidates  | Selection-only lock differences appear recoverable    | Prove exact historical equivalence and original authority; preserve original recipe/source identities              |
| Two ACE recipes           | Real prompt/lyrics/meter/duration changes             | Recover genuinely matching new evidence or generate the actual changed recipe; never pad audio or widen tolerances |
| Six Qwen Edit entries     | Residual workflow-lock discrepancy unresolved         | Compare original executed graph, adapter settings and current selection semantically                               |
| Two legacy Qwen bundles   | Incomplete complete-model-set evidence                | Recover original authoritative set or retain the hold; do not drop inconvenient duplicates                         |
| Thirteen video identities | Decoder/AV identity mismatch; derivative cases differ | Compare encoded bytes, original decoder/version/raw output and actual frame/AV identity before changing validation |

Frame-checksum headers can include software/timing metadata, but header drift has
not been established as the sole cause of these original discrepancies. Preserve
all failed hashes and rights holds until evidence supports a scoped correction.
The maintained evidence-repair command is not permission to attach today's settings
to yesterday's output.

**Exit:** every repaired row has original evidence justifying the repair and an
immutable before-copy; changed recipes/media obtain new execution and review.
`gallery:verify`/coverage results state remaining rows accurately. New public media
still needs explicit output-specific publication approval.

## B3 — close the remaining selected FLUX3 templates and visual holds

**Native/visual/rights acceptance, not one generic source bug. Budget: 2–4 hours
for L2/L3's finite controls; 4–8 hours for an initial quality-diagnosis tranche.
Publication timing depends on approval and successful new evidence.**

The original eight-template target expanded with NASA/Monet candidate work and
controls. Keep public templates, private planning candidates, experimental
canonical graphs and scale-zero controls distinct; do not inflate the published
catalog count. A generated private trial is not a registered public recipe.

Retained original visual holds include erased lower packaging panels, incorrect
stem/control counts or unintended labels, erased central accents, and unconfirmed
NASA styling. Windows two-reference quality holds remain separate from Linux
results. The two Windows outputs completed but did not satisfy their frozen visual
brief. No dropped-reference code cause was established in that audit.

**Approach:** use the exact reviewed model/adapter/input order and original prompt,
size, steps, seed and resource recipe. Compare consumed controls and actual output
against the specific visual criterion. After two equivalent failures, change one
justified engineering/recipe hypothesis or retain the quality hold. A modified
recipe needs a new lock and new evidence. A Linux no-offload output cannot qualify
the Windows CPU-offload variant or vice versa.

NASA requires a convincing intended style decision; Monet's scale-zero comparison
showed only a limited brushwork difference. Public recipe acceptance and concrete
rights/output-publication approval remain open. Preserve the original Tarot hold;
testing Monet did not grant Tarot's missing permission.

## B4 — full advertised image/prototyping breadth

**Some implementation exists; complete native variant/transition coverage does
not. Budget: 0.5–1 day for a representative additional transition tranche; several
days for the wider advertised route set after inventory and model access.**

**Where:** BE executable specifications, image adapters, native modular selection
and custom composition; FE operation catalog/change planner, block persistence,
template admission, developer-first UX and service export.

Freeze the union of actual advertised models/tasks/variants across palette,
starters, templates, pickers and documentation. Compare to the exact reviewed
upstream native exports. A missing MoDiff adapter is not an upstream exception.
For a legitimate whole-pipeline exception, document the exact task, unavailable
stage editing/reuse, supported controls and revisit trigger.

Prioritize distinct semantic transitions rather than every alias permutation:
conditioning/latent changes, edit versus img2img, added required media, guidance
contract changes, custom nodes and saved interfaces. Retain edited values and wires,
review destructive changes, and prove actual execution after save/reinsert. Expand
audio/video usability independently; September's image announcement did not claim
that video simplification was completed. Standalone Python-code generation remains
a separate product feature, not a promise made by the current service exporter.

## B5 — missing catalog executions and resource recipes

**Large unmeasured campaign. Budget: first 2 hours to recover/classify receipts and
measure a representative runnable batch; thereafter estimate from actual runtimes
and artifact availability. The earlier extra 5–10-working-day allowance is
provisional, not a commitment.**

Dependencies are B1 for unsupported canonical scopes, model/access/license/input
availability, usable target hardware and complete current evidence capture.
The 58 resource tuples are 23 resident, 26 model-CPU, eight group-disk and one
group-CPU in the retained census. They are not every-template × every-mode ×
every-device. Preserve the maintained representative-selection rules and exact
workload identities.

**Approach:**

1. Recover original complete receipts from their originating hosts. Absence in the
   current local registry does not prove absence everywhere. A custom-artifact
   exemption is not a measured model pass.
2. Refresh source contracts and freeze the remaining exact row set. Resolve model,
   component, input and recipe pins before downloading/running anything.
3. Group runnable rows by model for scheduling; each different locked workload
   still needs its own evidence. Run one model job per host with task-specific
   time/cancellation bounds. Do not launch an unbounded all-catalog loop.
4. Collect actual outputs, durations/peaks, reuse/resource behavior and task-specific
   quality. Unknown or missing measurements stay missing. Keep supported resources
   and hardware observations distinct from broad Auto authority.
5. Re-estimate after the first batch using remaining counts, observed durations,
   downloads, failures and review time. Stop at the agreed finite batch boundary.

Maintained commands include `release:qualification:run`, `release:resources:run`,
`release:template:qualify`, `release:hardware:qualify` and contract/report generators.
Read their current CLI help/source before invoking. Some commands called dry-run
still write derived reports. No general all-203 native campaign should be claimed
until its supported collector scope actually exists.

## B6 — hardware and optional compiler qualification

**External availability plus platform-specific work. Budget: about 0.5–1 day per
new available hardware profile for setup and a finite initial qualification;
model breadth, driver problems and downloads add time. No ETA for unavailable
hardware.**

- Windows NVIDIA and CPU can supply their own exact native results.
- Linux Radeon shared memory supplies conditional Linux observations; it is not
  additional dedicated VRAM and does not qualify a 16 GB Windows GPU.
- Apple Silicon is unavailable. The mandatory MPS predicate remains blocked.
- Intel XPU, AMD Windows and MI300X physical access is unestablished. AMD Windows
  also lacks the fully reviewed SDK wheel lock required by its current installer.
- Optional Windows compilation needs a compatible reviewed toolchain and actual
  kernel probe. Eager operation without Triton is independently useful and must
  remain usable. Do not invent a Triton-Windows pin or install arbitrary wheels.

The product's runtime tiers and per-model hardware support statements need
consistent wording. Changing a claimed support scope is a deliberate release
decision, never a way to silently pass a missing physical-hardware gate.

## B7 — publication, final release and durable handoff

**Budget: 2–4 hours once outputs, rights and all required gates are ready; approval
waits and new failures are outside that estimate.**

Obtain concrete approval for each proposed new output and its rights/provenance.
Publish approved media to the configured public Hugging Face Dataset, pin the
actual revision, verify bytes/availability and update only approved source
descriptors/hashes/docs in Git. Never publish private logs, input media or generated
files merely because account write access is available.

Freeze the final source/evidence pair. Run complete backend and compatible client
acceptance, asset and qualification gates. Failed, blocked and short-circuited
stages remain explicit. Deploy the verified bundle, preserve user data and produce
a supported-workflow/limitations handoff with actual source/runtime identity.

The 28 October Windows candidate can be useful and testable while these rows stay
open. It must not be represented as their completion.
