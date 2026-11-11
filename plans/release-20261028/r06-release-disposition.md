# R06 release disposition and remaining work

The bounded R06 audit is complete. **The development candidate is available for
manual testing; the advertised public release remains blocked.** Passing CI,
accepted private media and permission to merge or publish are separate decisions.
No model, download, installation, asset publication, deployment or source refresh
was performed for this audit. All 43 missing-model cases remain user-held.

## Source and latest CI at the audit checkpoint

The audited heads are backend `cb648e5120362c0b8f9aeeaaab86ddbe38640c3e` and client
`31ccd3bd2f9230477ad1128462149a9c6de3c5f5`, in the same two open, unmerged PR7s.
Both latest PR runs completed successfully:

- [Backend CI](https://github.com/sdevil7th/MoDiff/actions/runs/42446917882): lint
  and all three CPU platform jobs passed.
- [Client CI](https://github.com/sdevil7th/MoDiff-client/actions/runs/42446922023):
  quality, complete browser quality and both platform smoke jobs passed.

Both PRs were mergeable, were not drafts, and had no recorded review submissions
or inline review threads. That is not a human approval or a universal hardware
qualification. This audit establishes no new CI failure or merge conflict.
Subsequent documentation commits have their own hosted checks; the eight-job
result above stays bound to the actual audited heads.

The deployed application remains backend `e06a4f16` with the unchanged `f6cddacc`
client product. Read-only checks returned ready health and HTTP 200 at the
[manual-test app](http://127.0.0.1:8298/); its served index exactly matched the
canonical packaged index. No running process was changed.

## Fresh release checks

Eight finite, local checks ran with the existing tools and test interpreter.
All prior asset files, manifests, rights records, packaged builds and original
release-report inputs were preserved exactly.

| Check                                                               | Actual result                                                                                                          | Meaning                                                                                                                              |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `node scripts/template-gallery-harness.mjs verify`                  | Blocked: the same 74 errors as the retained report, across 63 of 80 examples                                           | Current locks/evidence still disagree; not 74 missing files or required model jobs.                                                  |
| `node scripts/template-gallery-coverage.mjs`                        | Blocked: 17 asset-backed of 89 supported templates; one additional user-supplied template; 72 uncovered supported rows | Coverage requires matching evidence, not merely a visible file. Ten of 17 video templates lack an accepted motion-preview record.    |
| `node scripts/workflow-library-verify.mjs`                          | Passed: 203 supported and six qualified experimental portable workflows                                                | Static graph/layout/contracts, not 209 fresh model executions.                                                                       |
| `node scripts/release-contract.mjs --check`                         | Blocked: retained report is stale                                                                                      | Current calculation adds a contact-sheet prompt mismatch; it does not promote another row.                                           |
| `node scripts/resource-qualification-report.mjs --check`            | Passed: retained resource report is reproducible                                                                       | Its status remains incomplete: zero of 60 registered resource tuples qualified in this report. Reproducibility is not qualification. |
| `node scripts/release-candidate-report.mjs --check --require-ready` | Blocked by five gate categories                                                                                        | The private fresh projection independently retains all five categories.                                                              |
| `node scripts/template-asset-release-contract.mjs`                  | Passed for the existing built shell and pinned approved-subset source                                                  | No new build, remote verification, rights grant or upload.                                                                           |
| `python scripts/template-gallery-assets.py verify`                  | Blocked: complete local inventory differs                                                                              | Four held files are absent; 15 available records have metadata-only differences. Byte hashes remain exact.                           |

The full `check:acceptance` chain was not repeated: its independently inspected
release gates are blocked, and repeating unrelated build/browser/model work would
not supply missing evidence. Existing complete product gates retain their actual
source scope. Every command, exit status, bounded original log and digest is kept
in the ignored private R06 evidence directory.

## Gallery and asset findings

The 74 Gallery errors consist of **63 recipe locks, three prompt locks, two audio
duration mismatches and six provenance errors**. Categories overlap rows.

The old public continuation is 15 seconds against a 90-second recipe; the old
text-to-audio example is 30 seconds against a 75-second recipe. The two affected
legacy Qwen examples each retain a failed exactness comparison and two
complete-model-set hash errors. Changing a lock, adding a missing model from the
current catalog, padding audio or widening tolerances would not repair original
execution evidence.

The complete asset manifest declares **395 files /491,202,898 bytes**. Its rights
ledger marks **391 approved and four held**. Every available file matches its
declared byte length and SHA-256. The absent set exactly matches the four declared
withheld ACE Chinese New Year/Ghibli media files. Recorded redistribution
permission is still required for those four; existing partial-source support
does not approve them.

The complete local inventory check also calculates different MIME type or optional
width/height metadata for 15 records. These differences change no file bytes,
sizes, paths or purpose classifications. They need a deterministic, cross-platform
metadata diagnosis before any contract change. Do not regenerate the approved
asset-set identity or rights ledger merely to silence this check. The pinned
approved-subset source/build check passes separately. Documentation's obsolete
360/356 counts have been corrected to the actual 395/391 contract counts.

## Current release projection and proof limits

The existing metadata producers were run with only narrowly recorded private
input/output path bindings. Original canonical reports were not overwritten.
The current contract adds `prompt_settings_hash_mismatch` for
`z_image_cinematic_contact_sheet`, alongside its existing lock/workflow mismatch.
Only that reason, its derived missing-evidence lists and the contract hash differ.
Eligibility remains unchanged:

| Release requirement                    | Current registered evidence                                                              |
| -------------------------------------- | ---------------------------------------------------------------------------------------- |
| Catalog template eligibility           | One of 90; 89 incomplete                                                                 |
| Complete qualification receipts        | One of 90; 89 incomplete                                                                 |
| Canonical workflow eligibility         | One of 203; 202 incomplete                                                               |
| Resource tuples                        | Zero of 60; all 60 remain deferred/missing in this registry                              |
| Advertised supported physical profiles | CPU, NVIDIA CUDA and Apple MPS lack accepted current receipts                            |
| Other hardware observation             | AMD ROCm Linux is observed; that does not qualify those three profiles or all AMD routes |

These are registry/evidence gaps, not proof that every row fails to execute.
They overlap with held artifacts, unsupported canonical-accounting scopes and
private native receipts. Private task audits, creative acceptance and large-host
success cannot be relabelled public qualification. The existing report field
`releaseContractCurrent` is not sufficient by itself: the separate freshness
check failed, so both the stale original and fresh projection are retained.

The cached-case campaign remains **five of nine privately accepted, four
unclosed**. Klein, rainy rail, ACE repaint, the photographic rocket with its
framing limit and the photographic Moonflower film with its stated limits stay
accepted. Boardwalk/car Auto technical delivery is closed; creative review is
separate. Wan retains its fixed-deadline zero-output hold. ACE melody variation
lacks required codecs. No accepted model, song, film or rocket is rerendered or
re-reviewed for this audit.

R04 still has external constrained-Windows/fresh-GPU-setup acceptance, a second
trained-artifact replacement gap and unavailable formal cached-adapter ownership
telemetry. R05 identifies an independent historical Vite emitter but not the
reviewer's exact original session. Those limits are not silently closed by R06.

## Ordered follow-up and closure criteria

The [first retained-evidence recovery tranche](gallery-evidence-recovery.md)
corroborates 58 previous-recipe records and preserves their authority in private
packets. It changes no current lock, public sidecar or qualification result.
The two missing-model-set bundles and three genuinely changed recipes remain
held under their own criteria.

1. **Maintain the merge/manual-test candidate.** Keep current code checks green
   and inspect the documentation heads' actual CI before merging. A maintainer
   reviews and decides on merge; this checkpoint does not perform it.
2. **Reconcile evidence before generation.** For each affected Gallery row,
   recover original creator/admission/consumed-input/task/media/model-set authority.
   Reuse a receipt only when it matches the exact declared scope. A historical
   selection-only equivalent can remain a previous-recipe example; it cannot
   automatically satisfy a current recipe. Preserve every original lock and failure.
3. **Repair maintainer metadata only after diagnosis.** Review the fresh contract
   projection and the inventory's MIME/geometry portability differences. Any
   tooling repair needs an original failing regression, preserved identities,
   focused tests and required paired gates. Never invent absent telemetry or
   promote qualification while refreshing hashes.
4. **Resolve human/access holds explicitly.** Retain the pending boardwalk/car
   creative disposition, Wan performance hypothesis and missing ACE capability.
   Obtain exact-file rights decisions and output-specific publication authority
   separately. Closed song/film/rocket decisions are not reasked.
5. **Admit only the still-missing executions.** Canonical accounting, resource and
   physical-platform gaps need separately scoped evidence after recovery. Keep
   all 43 missing-model cases held; SSD space grants no download/cache-move
   authority. Models stay serial on Linux with their original workload and guards.
6. **Recheck public readiness at a real release checkpoint.** Exact recipe/input/
   provenance/rights and physical-profile requirements must pass before public
   publication. Do not reduce advertised coverage, weaken checks, rewrite public
   assets or start another broad campaign merely to obtain green output.

## Bounded manual UI check

Use a separate QA draft at the local test app. Inspect Run-blocked recovery,
the selected-workflow Setup checklist, Edit stages/Expand block, the distinction
between reusable Blocks and Python nodes, supported same-model task changes,
Undo/Redo and Save/reopen. Browsing and setup inspection should install or run
nothing. Preserve accepted workflows and existing outputs. Model generation is
separate from these UI checks and remains Linux-only under normal admission.


## ACE Auto follow-up checkpoint

The [ACE current Auto checkpoint](gallery-evidence-recovery.md#ace-current-auto-execution-gaps--native-technical-closure)
closes the missing native Auto execution/reader bindings for the current 75-second
song and joined 90-second continuation. Two separately guarded ordinary UI Runs
completed on Linux with strict audits, full playback, normal releases and exact
preservation. The original accepted Expert recordings and absent old consumed
fields remain unchanged. New private proposals have zero recipe/media/provenance
errors; listening, formal template review and publication remain pending.

The narrow maintainer observation reader passes the complete client gate and
does not change Auto admission, the app bundle, backend runtime or qualification.
Public Gallery 74 and all broader release holds remain; this is not public
activation. All 43 missing-model cases remain user-held in the same two PR7s.

## Deterministic metadata and current-source verification — 2026-11-08

This checkpoint supersedes the earlier unresolved maintainer diagnosis and the
two new ACE candidates' pending-listening statements. It changes no public
Gallery entry, template review, qualification, rights or publication decision.

The inventory defect was reproduced against the original host-dependent MIME
lookup. Repository-owned mappings now keep supported formats deterministic,
including the existing `audio/x-wav` spelling; unknown extensions remain opaque.
PNG/WebP header validation and byte/size/purpose/complete-set checks remain strict.
The original failing regression and related malformed-image/case tests are
retained. All 40 asset tests pass. Seven WAV metadata discrepancies resolve.
All 391 available files retain their original bytes, sizes and purposes.

Eight older WebP descriptors still declare an opaque content type and omit
512-by-512 dimensions. A separate complete 395-record metadata-only proposal
changes only those fields. Independent full WebP decoding confirms all eight
512-by-512 records; their existing approved byte identities remain exact.
The original asset set, rights ledger, immutable Dataset pin and all four rights-held absences stay unchanged. The complete-set
check remains held; the proposal is not an approved/public manifest update.

The candidate reporter previously assigned `releaseContractCurrent` unconditionally.
It now runs the existing bounded read-only `release-contract.mjs --check`.
Staleness, verifier failure, timeout or unavailable execution makes the gate false
with a named blocker. It does not regenerate source or expose private diagnostics.
Three original failing CLI regressions now pass; 26 related report/contract/policy
tests pass. The actual current-source private projection returns false, retaining
every other gate, blocker, coverage count and original report unchanged. Correct
freshness reporting does not qualify a template or hardware profile.

Validation: `node --test scripts/release-candidate-report.test.mjs
scripts/release-contract.test.mjs scripts/release-qualification-policy.test.mjs`
passes 26 tests; `python -B scripts/template-gallery-assets.test.py` passes 40;
`npm run test:release-contract` passes 49 with six declared optional skips;
`ruff check scripts/template-gallery-assets.py scripts/template-gallery-assets.test.py
--select E9,F` passes. The complete final `npm run check` passes in 506.09
seconds, with exact source fingerprints retained before/after. All 638 frozen release,
asset and build inputs, including the 119-file product bundle, remain exact.
Backend executable code is unchanged: its completed 9140-test/13202-subtest gate
retains its original scope. No native model, deployment or closed proof is repeated.

The user explicitly accepted both new Auto outputs, `udHS8txXiAwZ` and
`kjDOnhLsP6hu`, for private listening. Verdicts bind their exact original WAVs.
Earlier recordings and capture-time observations remain unchanged. Formal rubric,
exact-file rights and publication are still separate.

After actual upstream HTTP recovery, only the remaining failed setup job was
retried. All 16 push/PR checks passed at prior heads `61778e22` / `52a52ead`;
both PRs were clean/mergeable with no unresolved review threads. New tooling
commits require their own actual hosted outcomes; old green checks do not qualify
new heads. Changes stay in the same two existing PR7s.

### Remaining work and closure criteria

1. Bind the technically reviewed eight-record metadata proposal to dependent
   source/rights identities in a separately authorized manifest/Dataset update.
   No public update is part of this work. Four absent rights-held assets
   require exact-file redistribution decisions; do not invent or replace them.
2. Complete formal contact-sheet and ACE template review, then obtain separate
   output-specific rights/publication decisions. Private acceptance is recorded;
   accepted outputs need no rerun or repeated listening request.
3. Continue evidence recovery for current Gallery rows without rewriting their
   original locks. The retained 74-error report, two Qwen bundles lacking original
   complete-model-set authority and unavailable public qualification remain held.
4. Keep boardwalk/car creative disposition separate from closed Auto technical
   proof, Wan's fixed-deadline zero-output hold and ACE melody-codec absence.
   R04 constrained Windows/fresh GPU setup and artifact/telemetry limits, plus
   R05's unbound original reviewer session, retain their stated proof boundaries.
5. Physical CPU/NVIDIA/MPS and resource-tuple evidence require separately scoped
   admission and actual supported-machine proof. Do not infer it from unit CI or
   the large AMD host. All **43 missing-model cases remain user-held**.

The SSD remains available storage only. No downloads, model moves, installations,
uploads, cache/source/runtime changes or new model campaign occurred. Software
merge readiness and public release readiness remain distinct.
