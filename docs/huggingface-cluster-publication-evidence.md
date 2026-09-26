# Hugging Face Cluster publication evidence

This document defines the fail-closed publication boundary for a registered
Hugging Face Cluster route. It is deliberately separate from graph admission
and from the volatile receipt that allows one run on the current machine.

## Independent authorities

One route can be graph-qualified and draggable without being publicly
executable, Auto-eligible, or Gallery-eligible. These claims require separate
evidence:

1. `insertable` comes from the exact graph adapter, Studio execution
   specification, artifact revision, and registered `BlockDefinitionV2` route.
2. `liveProof` comes only from an approved, route-specific promotion receipt.
   A schema-v2 receipt binds the definition, admission, exact registered
   `BlockDefinitionV2` content hash and canonical SHA-256, artifact revision,
   Studio spec, generated media hash, frontend lifecycle, and workspace-owner
   decision.
3. `executable` is not granted by the output receipt. Each run must still pass
   current artifact, optional-runtime, dependency, and resource checks. The
   immutable catalog remains `executable: false` until a separate public
   runtime authority is designed and reviewed.
4. `autoEligible` additionally requires an exact-route resource measurement
   and an explicit Auto publication authority.
5. `galleryEligible` additionally requires an immutable published asset receipt
   and separate rights/publication review. Local approved review media is not a
   Gallery publication.

The promotion ledger therefore intentionally permits only `liveProof: true`;
it rejects any receipt that also tries to enable runtime, Auto, or Gallery.
Model-capability records are family-level discovery/runtime metadata and are
not another publication authority. In particular, a family-level `liveProof`
field must not be turned on because one workflow in that family was approved;
the route-specific admission and this audit are the authoritative projection.

## Output-review receipt identity

The active ledger is
`data/huggingface-cluster-promotion-receipts.v2.json`. A current receipt must
contain this exact Block identity in addition to its existing admission,
artifact, Studio-spec, media, lifecycle, and owner-review bindings:

```json
{
  "blockDefinition": {
    "definitionId": "diffusers.cluster-admission:Pipeline:workflow:mode:mode",
    "contentHash": "block-definition-v2-1234abcd",
    "canonicalSha256": "sha256:<64 lowercase hex characters>"
  }
}
```

The backend compares that object with its current registered V2 pin every time
it resolves a promotion. A changed graph, boundary, controls, defaults, or
other canonical definition content therefore invalidates the approval instead
of silently carrying `liveProof` onto a different Block.

The earlier schema-v1 Qwen and MiniMax approvals are preserved unchanged in
the v1 ledger and copied into the v2 ledger's `historicalReceipts` section.
They remain useful review history, but they are explicitly non-authorizing
because they did not record the exact Block V2 content hash and canonical
SHA-256. The current `receipts` collection is empty until the workspace owner
reviews a fresh exact-route output; no approval was inferred during migration.

### Staging a new owner-approved receipt

`scripts/stage_huggingface_cluster_promotion_receipt.py` is the deterministic,
non-authorizing receipt builder. It accepts one current frontend
`run-provenance.json` plus the exact generated media file. It streams and
checks the media SHA-256 and byte size, requires the provenance route binding
to equal the current backend Block pin, and requires every lifecycle and owner
approval assertion as an explicit command flag. It never invents a timestamp,
review comment, approval, or lifecycle claim, and it never edits the checked-in
ledger.

Publication itself changes `publication.liveProof` from false to true, which
changes the manifest content hash embedded in the registered Block source.
The receipt must therefore bind the post-promotion Block identity, while its
evidence must still bind the current Block that actually ran. The staging flow
is explicitly two-phase and never mutates either identity.

First, from `MoDiff-client`, compile the non-installing post-promotion route
candidate:

```bash
MODIFF_POST_PROMOTION_ADMISSION_ID='<exact admission ID>' \
MODIFF_GENERATE_ROUTE_CANDIDATES=1 \
MODIFF_ROUTE_CANDIDATES_ONLY=1 \
MODIFF_AUDIT_DISCOVERY=1 \
MODIFF_ROUTE_CANDIDATE_OUTPUT='<new post-promotion report>.json' \
node --test scripts/registered-block-v2-route-audit.test.mjs
```

This uses a context-local backend projection that changes only the selected
admission's candidate `liveProof` value. It does not load a receipt, edit a
route pin, or write backend data. The normal strict 90-route audit remains a
separate run without these environment variables.

After the workspace owner has actually reviewed the exact output, return to
`MoDiff` and stage one atomic candidate bundle:

```bash
./.venv/bin/python scripts/stage_huggingface_cluster_promotion_receipt.py \
  --admission-id '<exact admission ID>' \
  --provenance '<review run>/run-provenance.json' \
  --media '<review run>/<generated asset>' \
  --post-promotion-candidate-report '<new post-promotion report>.json' \
  --receipt-id 'cluster-promotion:<route>:<review-date>' \
  --reviewed-at '<ISO-8601 time with UTC offset>' \
  --review-comment '<workspace owner decision for this exact asset>' \
  --approve-as-workspace-owner \
  --confirm-inserted-through-frontend \
  --confirm-expanded-official-blocks \
  --confirm-edited-parameters \
  --confirm-saved-and-refreshed \
  --confirm-values-restored-exactly \
  --confirm-collapsed-expanded-graph-equivalent \
  --confirm-generated-through-frontend \
  --atomic-bundle \
  --output '<new atomic promotion bundle>.json'
```

The output path must not already exist. The resulting file is still only a
candidate. It contains the current evidence route binding, independently
compiled post-promotion manifest and Block pins, schema-v2 receipt, complete
promotion-ledger candidate, and—when the admission has one—the rehashed
historical compiler-mapping ledger candidate. Review and apply those dependent
changes atomically to the promotion ledger, client registered route, backend
runtime pin, exact tests, and dependent mapping; never copy the outer bundle
itself over one of those files. Then run the normal strict route and publication
audits. Omitting any confirmation or the explicit workspace-owner approval
fails without producing a bundle. Old provenance whose manifest, artifact,
Studio spec, or Block V2 identity no longer matches the current evidence route
also fails; wall-clock recency is not used as a substitute for exact identity.

## Resource-evidence identity

`resource-recipe-coverage.v1.json` keeps two intentionally separate lanes.
`recipes` groups measurements by model type, dtype, offload mode, and
quantization mode for release-wide coverage. Several modes in one family can
share that tuple, so this lane is never route authority. Historical Auto
measurements and older receipts are preserved there with
`bindingStatus: legacy_unbound`; they are not rewritten or promoted.

`routeQualifications` contains only new receipts whose retained frontend proof
carried an exact `routeBinding`. The audit refuses to infer that identity from
a candidate name, template, or model type. A binding must exactly equal all of:

- the admission ID;
- the registered `BlockDefinitionV2` definition ID and content hash;
- the collision-resistant SHA-256 of the canonical `BlockDefinitionV2`;
- Studio execution-spec ID, content hash, and execution-profile ID;
- artifact repository and immutable revision;
- the admission's complete, canonically sorted model-dependency IDs, kinds,
  repositories, and immutable revisions.

The route-binding hash and report hash cover this object. The audit then
re-resolves the current backend-pinned V2 definition identity before it counts
the receipt. A stale, unknown, or sibling-workflow binding fails report
verification; its underlying measurement may be retained only in the
legacy/unbound family-evidence lane and qualifies no current Cluster route.

The normalized binding has this exact shape (values shown schematically):

```json
{
  "schemaVersion": 1,
  "admissionId": "diffusers.cluster-admission:Pipeline:workflow:mode:mode",
  "blockDefinition": {
    "definitionId": "diffusers.cluster-admission:Pipeline:workflow:mode:mode",
    "contentHash": "block-definition-v2-1234abcd",
    "canonicalSha256": "sha256:<64 lowercase hex characters>"
  },
  "studioExecutionSpec": {
    "id": "pipeline:mode:v1",
    "contentHash": "studio-spec-v1-12345678",
    "executionProfileId": "pipeline:profile"
  },
  "artifact": {
    "repository": "owner/repository",
    "revision": "<40 lowercase hex characters>"
  },
  "modelDependencies": [
    {
      "id": "text_encoder",
      "kind": "text_encoder",
      "repository": "owner/dependency",
      "revision": "<40 lowercase hex characters>"
    }
  ]
}
```

Unknown keys, mutable revisions, short/public-only V2 hashes without the
canonical SHA-256, copied sibling specifications, and a definition ID that is
not the admission ID all fail closed.

The retained live proof also locks the complete executed model-set hash and
includes the route-binding hash in its recomputed proof lock. Report generation
reopens the retained proof, verifies its file hash, proof lock, model set,
workload, recipe, task, graph, runtime, output, duration, and measured peak
memory against the receipt, and recomputes the recipe and report hashes. A
hand-edited receipt or report cannot become route evidence merely by copying a
route ID.

Legacy schema-v1 reports without `routeQualifications` remain readable as an
empty route lane. Newly generated reports always write the field explicitly.
The current 12 family measurements are all `legacy_unbound`, and the current
checked report contains zero exact route qualifications.

Even with that binding, `autoEligible` stays false until a distinct checked-in
Auto publication authority enables the route.

## Current audited routes

The checked-in evidence on September 2, 2026 resolves as follows:

| Route | Insertable | Live proof | Executable | Auto | Gallery | Release eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen Image 2512 Modular text-to-image | yes | no | no | no | no | no |
| MiniMax Music 3 official Modular workflow | yes | no | no | no | no | no |
| Wan 2.2 TI2V 5B text-to-video | yes | no | no | no | no | no |
| Cosmos 3 Nano Modular text-to-image | yes | no | no | no | no | no |
| Cosmos 3 Nano Modular text-to-video | yes | no | no | no | no | no |

The precise remaining blockers are:

- Qwen: fresh owner approval bound to the current Block V2 identity, per-run
  runtime authority, exact-route resource binding, immutable Gallery asset
  publication, and Gallery rights approval.
- MiniMax Music 3: fresh owner approval bound to the current Block V2 identity,
  per-run runtime authority, a declared and measured release resource recipe,
  immutable Gallery asset publication, and Gallery rights approval.
- Wan 2.2 TI2V 5B: per-run runtime authority, workspace-owner output approval,
  exact-route resource binding, immutable Gallery asset publication, and
  Gallery rights approval.
- Cosmos 3 Nano: the public model snapshot is pinned at
  `7a312c868bcce8e40b3eb40861300a9d0ba3fde1`, but both routes require the
  mandatory gated `nvidia/Cosmos-Guardrail1` revision
  `d6d4bfa899a71454a700907664f3e88f503950cf`. The workspace owner has not
  acknowledged that guardrail's NVIDIA terms, its exact `cosmos-guardrail`
  0.3.1 runtime has not been admitted, and neither the 34,986,890,561-byte
  Nano snapshot nor the 7,171,449,905-byte guardrail snapshot has completed a
  real resource/output qualification. No quantization, offload, executable,
  Auto, template, Gallery, or public flag is inferred from static graph
  admission.

The global release candidate remains blocked separately by incomplete template,
workflow, qualification-receipt, resource-recipe, and physical-profile evidence.
Those release-wide blockers do not revoke graph insertion, and graph insertion
does not bypass them.

## Audit command

Run:

```bash
./.venv/bin/python scripts/audit_huggingface_cluster_publication.py
```

The command validates both checked-in report hashes, confirms that the resource
and release reports bind the same release contract, re-resolves each exact
schema-v2 promotion receipt against the current Block pin, reports preserved
schema-v1 approvals as `historical_non_authorizing`, detects any mismatch
between a current receipt and `liveProof`, and emits route flags plus
machine-readable blocker codes. It performs no model load, download, graph
mutation, or workflow migration.
