"""Classify the 77 public templates without re-running graphs.

The release contract already records which historical Gallery examples still
match the current generic-node graphs. This module turns that evidence into a
keep / canary / first-example plan. It never submits a graph, mutates Studio
templates, or publishes Dataset media.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from modiff.local_review_receipts import canonical_content_hash


SCHEMA_VERSION = 1
RELEASE_CONTRACT_PATH = Path("data/release-contract.v1.json")
LOCAL_RECEIPT_LEDGERS = (
    Path("data/qualification/local-review/technical-candidates.v1.json"),
    Path("data/qualification/local-review/image-stitch-candidate.v1.json"),
)
CURRENT_STATUS = "legacy_node_contract_match"
STALE_STATUS = "stale"
BUCKET_CURRENT = "current_keep"
BUCKET_STALE = "stale_canary_required"
BUCKET_NEVER = "never_had_example"
_BOUNDARY = {
    "blanketRerunForbidden": True,
    "graphsSubmitted": False,
    "galleryMutated": False,
    "datasetPublished": False,
    "publicTemplateMutated": False,
    "qualificationHostRequiredForCanaries": True,
}


class PublicTemplateReconciliationError(ValueError):
    """Raised when the public-template reconciliation cannot be classified fail-closed."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise PublicTemplateReconciliationError(f"Reconciliation source is missing or linked: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PublicTemplateReconciliationError(f"Reconciliation source is not an object: {path}")
    return payload


def _local_receipt_workflow_ids(root: Path) -> set[str]:
    found: set[str] = set()
    for relative in LOCAL_RECEIPT_LEDGERS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for receipt in payload.get("receipts") or []:
            workflow_id = receipt.get("workflowId")
            if isinstance(workflow_id, str) and workflow_id:
                found.add(workflow_id)
    return found


def classify_public_template(template: dict) -> dict:
    if not isinstance(template, dict):
        raise PublicTemplateReconciliationError("Public template record must be an object.")
    template_id = template.get("id")
    workflow_id = template.get("canonicalWorkflowId")
    if not isinstance(template_id, str) or not template_id:
        raise PublicTemplateReconciliationError("Public template is missing an id.")
    if not isinstance(workflow_id, str) or not workflow_id:
        raise PublicTemplateReconciliationError(f"Public template {template_id} is missing canonicalWorkflowId.")

    historical = template.get("historicalEvidence")
    last_run = template.get("lastSuccessfulRealRun")
    missing = list(template.get("missingReleaseEvidence") or [])
    reasons = list((historical or {}).get("reasons") or []) if isinstance(historical, dict) else []
    missing_nodes = list((historical or {}).get("missingBackendNodes") or []) if isinstance(historical, dict) else []

    if historical is None:
        bucket = BUCKET_NEVER
        action = "plan_first_example_canary_on_qualification_host"
        note = (
            "No historical Gallery example exists. Qualification-field completeness is not a current "
            "generic-node example. Author the first example on a qualification host; do not treat this as a rerun."
            if not missing
            else "No historical Gallery example exists. Plan a first-example canary on a qualification host."
        )
    elif not isinstance(historical, dict) or not isinstance(historical.get("status"), str):
        raise PublicTemplateReconciliationError(f"Public template {template_id} has malformed historicalEvidence.")
    elif historical["status"] == CURRENT_STATUS:
        if not isinstance(last_run, dict):
            raise PublicTemplateReconciliationError(
                f"Public template {template_id} is marked current without lastSuccessfulRealRun."
            )
        bucket = BUCKET_CURRENT
        action = "keep_historical_example"
        note = (
            "Historical example still matches the current generic-node contract. Keep the bytes. "
            "Do not rerun unless a later lock drift appears."
        )
    elif historical["status"] == STALE_STATUS:
        bucket = BUCKET_STALE
        action = "plan_canary_rerun_on_qualification_host"
        note = (
            "Historical Gallery bytes remain; the generic-node lock does not match. "
            "Plan a canary rerun on a qualification host. Do not blanket-rerun the 77."
        )
    else:
        raise PublicTemplateReconciliationError(
            f"Public template {template_id} has unsupported historicalEvidence.status {historical['status']!r}."
        )

    return {
        "templateId": template_id,
        "label": template.get("label"),
        "canonicalWorkflowId": workflow_id,
        "modelType": template.get("modelType"),
        "mode": template.get("mode"),
        "bucket": bucket,
        "action": action,
        "doNotRerunHere": True,
        "releaseEligible": bool(template.get("releaseEligible")),
        "historicalEvidenceStatus": None if historical is None else historical.get("status"),
        "staleReasons": reasons,
        "missingBackendNodes": missing_nodes,
        "missingReleaseEvidence": missing,
        "reviewedExampleQualityReviewStatus": template.get("reviewedExampleQualityReviewStatus"),
        "reviewedExampleVerificationStatus": template.get("reviewedExampleVerificationStatus"),
        "lastSuccessfulRealRunCapturedAt": last_run.get("capturedAt") if isinstance(last_run, dict) else None,
        "lastSuccessfulRealRunVerificationStatus": (
            last_run.get("verificationStatus") if isinstance(last_run, dict) else None
        ),
        "note": note,
    }


def build_public_template_reconciliation(root: Path, *, release_contract: dict | None = None) -> dict:
    root = root.resolve()
    document = release_contract if release_contract is not None else _load_json(root / RELEASE_CONTRACT_PATH)
    templates = document.get("templates")
    if not isinstance(templates, list) or not templates:
        raise PublicTemplateReconciliationError("Release contract has no public templates to reconcile.")

    items = [classify_public_template(template) for template in templates]
    ids = [item["templateId"] for item in items]
    if len(set(ids)) != len(ids):
        raise PublicTemplateReconciliationError("Release contract contains duplicate public template ids.")

    local_workflows = _local_receipt_workflow_ids(root)
    public_workflows = {item["canonicalWorkflowId"] for item in items}
    overlap = sorted(local_workflows & public_workflows)
    if overlap:
        raise PublicTemplateReconciliationError(
            "Local technical receipts overlap public templates; reconcile before claiming a disjoint campaign."
        )

    counts = Counter(item["bucket"] for item in items)
    reason_counts = Counter()
    for item in items:
        if item["bucket"] != BUCKET_STALE:
            continue
        reason_counts["+".join(item["staleReasons"]) if item["staleReasons"] else "unspecified"] += 1

    ledger = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "public_template_reconciliation",
        "boundary": dict(_BOUNDARY),
        "source": {
            "path": str(RELEASE_CONTRACT_PATH),
            "coverage": document.get("coverage"),
        },
        "policy": {
            "keepCurrentHistoricalExamples": True,
            "canaryOnlyStaleAndNeverHadExample": True,
            "blanketRerunForbidden": True,
            "executeOnThisHost": False,
        },
        "items": items,
        "overlapWithLocalTechnicalReceipts": overlap,
        "classifiedAt": _now(),
        "summary": {
            "templateCount": len(items),
            "currentKeepCount": counts[BUCKET_CURRENT],
            "staleCanaryCount": counts[BUCKET_STALE],
            "neverHadExampleCount": counts[BUCKET_NEVER],
            "staleReasonCounts": dict(sorted(reason_counts.items())),
            "localTechnicalReceiptOverlapCount": len(overlap),
        },
    }
    expected_current = (document.get("coverage") or {}).get("currentHistoricalEvidenceCount")
    expected_historical = (document.get("coverage") or {}).get("historicalEvidenceCount")
    if isinstance(expected_current, int) and expected_current != counts[BUCKET_CURRENT]:
        raise PublicTemplateReconciliationError(
            "current_keep count drifted from release-contract coverage.currentHistoricalEvidenceCount."
        )
    if isinstance(expected_historical, int) and expected_historical != counts[BUCKET_CURRENT] + counts[BUCKET_STALE]:
        raise PublicTemplateReconciliationError(
            "current+stale count drifted from release-contract coverage.historicalEvidenceCount."
        )
    ledger["contentHash"] = canonical_content_hash(ledger)
    return ledger


def render_public_template_reconciliation_html(ledger: dict) -> str:
    sections = []
    titles = (
        (BUCKET_CURRENT, "Keep (do not rerun)"),
        (BUCKET_STALE, "Stale canary (qualification host only)"),
        (BUCKET_NEVER, "Never had an example (first-example canary)"),
    )
    for bucket, title in titles:
        rows = [item for item in ledger["items"] if item["bucket"] == bucket]
        cards = []
        for item in rows:
            extra = ""
            if item["staleReasons"]:
                extra = f"<p>Reasons: {', '.join(item['staleReasons'])}</p>"
            if item["missingBackendNodes"]:
                extra += f"<p>Missing historical nodes: {', '.join(item['missingBackendNodes'])}</p>"
            cards.append(
                "<article>"
                f"<h3>{item['templateId']}</h3>"
                f"<p>{item['label']}</p>"
                f"<p><code>{item['canonicalWorkflowId']}</code></p>"
                f"<p>{item['note']}</p>"
                f"{extra}"
                "</article>"
            )
        sections.append(f"<h2>{title} — {len(rows)}</h2>" + "".join(cards))
    summary = ledger["summary"]
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\" />"
        "<title>MoDiff public template reconciliation</title>"
        "<style>body{font-family:sans-serif;max-width:960px;margin:24px auto;padding:0 16px}"
        "article{margin:0 0 20px;padding:0 0 16px;border-bottom:1px solid #ccc}"
        "code{font-size:12px}</style></head><body>"
        "<h1>Public template reconciliation</h1>"
        "<p>No graph was submitted. Keep the 26 current historical examples. "
        f"Plan canaries only for {summary['staleCanaryCount']} stale templates and "
        f"{summary['neverHadExampleCount']} templates that never had an example.</p>"
        + "".join(sections)
        + "</body></html>\n"
    )


def write_public_template_reconciliation(root: Path, destination: Path | None = None) -> dict:
    ledger = build_public_template_reconciliation(root)
    target = destination if destination is not None else root / "review-pending" / "public-templates"
    _write_json(target / "reconciliation.v1.json", ledger)
    (target / "index.html").write_text(render_public_template_reconciliation_html(ledger), encoding="utf-8")
    return ledger
