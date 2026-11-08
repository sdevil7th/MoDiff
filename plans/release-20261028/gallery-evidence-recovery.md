# Retained Gallery evidence recovery: first tranche

This tranche recovers **58 previous-recipe evidence records** without changing
their locks, provenance or media. **It repairs zero current-recipe bindings**:
the strict Gallery total remains 74 errors. An old execution cannot become a
current native graph merely by changing its recorded hash.

There are 57 packets with available, byte-verified media and one rights-held
descriptor packet. Originals remain intact. No model, download, installation,
public upload, runtime change or qualification promotion was performed. All 43
missing-model cases remain user-held.

## Current reconciliation

The audit covers all 80 published entries at backend `6a290b07` and client
`a9c25aca`. It reuses the byte-bound R06 Gallery verification at the same
unchanged executable source. Sidecar/media hashes, current/historical lookups,
creator settings and public-record validation have separate roles.

| Entries | Disposition                                                                                                       |
| ------- | ----------------------------------------------------------------------------------------------------------------- |
| 17      | Current lock and public record readable under the maintained Gallery rules; no new native/hardware qualification. |
| 58      | Previous-recipe lock and public evidence corroborated; original authority archived in recovery packets.           |
| 2       | Previous-recipe match but original complete model-set authority absent; provenance held.                          |
| 3       | Real creator changes: the two ACE audio recipes and Z contact-sheet prompt; no rebind.                            |

Of the 58 corroborated historical entries, 51 match after omitting only the
later execution selection. Seven also use the existing recorded immutable-Hub-
revision equivalence rule. All other artifact settings remain locked, and the
exact repository/revision must occur in the original published run. Neither
lookup supplies proof of the current selected execution graph.

The 63 recipe-lock errors therefore comprise 60 historical matches, including
two provenance holds, and three changed recipes. Classification does not waive
the strict current gate.

## Preserved recovery packets

The ignored recovery archive contains 58 per-template packets, an index and
121 content-addressed original JSON blobs. Each packet retains the original
manifest entry, provenance/review bytes and hashes, original task/capture and
graph/proof/model/input/source identities where present, and media paths/hashes
checked against the asset/rights records. Missing observations stay missing.

Current-versus-historical comparison is separate. Current qualification, new
quality approval and new publication approval are explicitly false. Original
copies and available media references were independently byte-checked. Packet
projections are distinguished from raw originals. No public sidecar, Dataset
pin, rights record or asset identity was rewritten.

| Packet lane            | Count | Boundary                                                     |
| ---------------------- | ----- | ------------------------------------------------------------ |
| Historical creator     | 54    | Original creator authority, not current graph proof.         |
| Historical derivative  | 3     | Original derivative and source-run bindings remain distinct. |
| Rights-held descriptor | 1     | Ghibli media stays absent/withheld; no new approval.         |

Derivative records are `ace_step_lyric_music_video`, `qwen_layered_portrait` and
`wan_22_ti2v_5b_seed_vault`. The lyric derivative's seed/audio/lyric controls
differ from its source run; its derivative authority is not attached to that
source as consumed settings. This does not reapprove old or rejected media.

Compared with the old 49-candidate triage, 46 retain corroborated historical
records. Twelve additional entries are corroborated: six Qwen Edit rows, four
LTX video rows, the lyric derivative and Wan TI2V derivative. The old contact-sheet
candidate now has a changed prompt; two legacy Qwen bundles remain provenance-
held. The recovered total is therefore 58, not 61 or 63.

## Remaining provenance and recipe holds

`qwen_low_vram_product_concept` and `qwen_low_vram_text_rendering` retain their
original schema-1 duplicate bundles. Both runs in each have a singular model
identity and three-node graph, but no `models` complete-set record. This is an
absent field, not a wrong hash that can simply be recomputed. The six provenance
errors remain.

Bounded filename searches checked existing Windows evidence roots and the known
Linux validation tree. No matching original output/model-fingerprint/event
sidecars were found within those bounds. Per-root/global caps and roots are
recorded; this does not prove no external archive exists. A graph naming one
loader and today's installed model list cannot supply missing original execution
authority. No synthetic model set or validator relaxation was added.

The ACE recipes still differ in prompt, lyrics, duration and meter: old text-to-
audio is 30 seconds against 75, and continuation is a 15-second tail against the
joined 90-second recipe. The contact-sheet prompt changed. Matching later
evidence needs actual creator, submitted/consumed controls, graph and output
identities. Accepted private candidates are not automatic public replacements.

## Verification and next action

Existing focused recovery/historical/provenance checks passed **37 tests**, zero
failures/skips, in 80.42 seconds:

```text
node --test --test-name-pattern="retained|historical|previous|provenance" scripts/template-gallery-runner.test.mjs scripts/template-operation-workflows.test.mjs scripts/template-gallery-harness.test.mjs
```

The actual invocation used a structured argument list. Checks retain rejection
of changed admission/creator values, wrong graphs/consumed values, task/node
mismatches, reordered inputs and runtime/model/output substitution. These
controls were not changed; existing full product gates retain their source scope.

Public schema-2 copies intentionally redact source details. Applying a private
unredacted validator directly to them produces expected redaction diagnostics;
the maintained public verifier uses pinned attestations. The recovery audit
keeps those diagnostics separate from actual Gallery errors. Original private
path/encoding/reader failures remain preserved; only metadata assembly was
corrected, with no model rerun.

1. Use the packets to reference/review previous recipes, without counting them
   as current-template executions or rewriting public locks.
2. Keep the legacy Qwen rows held until their original complete execution
   authority is recovered. No download or new model run is authorized for them.
3. Evaluate matching later receipts for the three changed recipes separately,
   preserving original human verdicts and publication limits.
4. Any public metadata/media repair requires exact authority and a concrete
   reviewed candidate before publication. Existing rights holds stay intact.

The [R06 disposition](r06-release-disposition.md) retains broader resource,
qualification, hardware, creative and publication blockers. Changes stay in
the same two PR7s; the current application and model/runtime state are unchanged.
