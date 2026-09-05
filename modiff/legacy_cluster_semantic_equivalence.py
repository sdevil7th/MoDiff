"""Checked-in semantic-equivalence authority for historical Cluster migration.

Historical Cluster manifests must never be rebound to a current catalog route by
name, model, or presentation shape.  This module validates a data-only review
ledger whose receipts bind the complete historical execution identity to one
exact current ``BlockDefinitionV2`` graph and public interface.  A receipt is
review authority only: per-instance compiler mappings, read-only preview,
explicit apply confirmation, exact backups, and rollback remain mandatory.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


SEMANTIC_EQUIVALENCE_RECEIPTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "legacy-cluster-semantic-equivalence-receipts.v1.json"
)
SEMANTIC_EQUIVALENCE_SCHEMA_VERSION = 1
_MAX_LEDGER_BYTES = 2 * 1024 * 1024
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_BLOCK_DEFINITION_HASH = re.compile(r"^block-definition-v2-[0-9a-f]{8}$")
_BLOCK_GRAPH_HASH = re.compile(r"^block-graph-v2-[0-9a-f]{8}$")
_BLOCK_INTERFACE_HASH = re.compile(r"^block-interface-v2-[0-9a-f]{8}$")
_RECEIPT_ID = re.compile(r"^legacy-cluster-equivalence:[A-Za-z0-9._:-]{1,180}$")
_BOUNDARY = {
    "checkedInReviewAuthority": True,
    "doesNotInferFromLabelsOrModels": True,
    "requiresExactCompilerMapping": True,
    "doesNotAuthorizeWorkflowMutation": True,
}


class LegacyClusterSemanticEquivalenceError(ValueError):
    """Raised when historical semantic-equivalence authority is malformed."""


def canonical_content_hash(value: Mapping[str, Any], *, omit: str = "contentHash") -> str:
    payload = deepcopy(dict(value))
    payload.pop(omit, None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _exact_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise LegacyClusterSemanticEquivalenceError(f"{label} has missing or unknown fields.")
    return value


def _text(value: Any, label: str, *, maximum: int = 4096, pattern: re.Pattern[str] | None = None) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or (pattern is not None and not pattern.fullmatch(value))
    ):
        raise LegacyClusterSemanticEquivalenceError(f"{label} is invalid.")
    return value


def _studio_spec(value: Any, label: str) -> dict[str, str]:
    item = _exact_object(value, {"id", "contentHash", "executionProfileId"}, label)
    return {
        "id": _text(item["id"], f"{label}.id", maximum=512),
        "contentHash": _text(item["contentHash"], f"{label}.contentHash", maximum=128),
        "executionProfileId": _text(item["executionProfileId"], f"{label}.executionProfileId", maximum=512),
    }


def _historical_binding(value: Any, label: str) -> dict[str, Any]:
    item = _exact_object(
        value,
        {
            "manifestDefinitionId",
            "libraryRevision",
            "manifestContentHash",
            "executionAdmissionId",
            "studioExecutionSpec",
            "executionGraphHash",
            "interfaceHash",
        },
        label,
    )
    return {
        "manifestDefinitionId": _text(item["manifestDefinitionId"], f"{label}.manifestDefinitionId", maximum=512),
        "libraryRevision": _text(item["libraryRevision"], f"{label}.libraryRevision", pattern=_COMMIT),
        "manifestContentHash": _text(item["manifestContentHash"], f"{label}.manifestContentHash", pattern=_SHA256),
        "executionAdmissionId": _text(item["executionAdmissionId"], f"{label}.executionAdmissionId", maximum=1024),
        "studioExecutionSpec": _studio_spec(item["studioExecutionSpec"], f"{label}.studioExecutionSpec"),
        # Historical schemas did not use BlockDefinitionV2 hash prefixes.  The
        # reviewer therefore pins collision-resistant canonical SHA-256 values.
        "executionGraphHash": _text(item["executionGraphHash"], f"{label}.executionGraphHash", pattern=_SHA256),
        "interfaceHash": _text(item["interfaceHash"], f"{label}.interfaceHash", pattern=_SHA256),
    }


def _destination_binding(value: Any, label: str) -> dict[str, Any]:
    item = _exact_object(
        value,
        {
            "manifestDefinitionId",
            "libraryRevision",
            "manifestContentHash",
            "executionAdmissionId",
            "blockDefinitionId",
            "blockDefinitionContentHash",
            "blockDefinitionCanonicalSha256",
            "executionGraphHash",
            "interfaceHash",
        },
        label,
    )
    return {
        "manifestDefinitionId": _text(item["manifestDefinitionId"], f"{label}.manifestDefinitionId", maximum=512),
        "libraryRevision": _text(item["libraryRevision"], f"{label}.libraryRevision", pattern=_COMMIT),
        "manifestContentHash": _text(item["manifestContentHash"], f"{label}.manifestContentHash", pattern=_SHA256),
        "executionAdmissionId": _text(item["executionAdmissionId"], f"{label}.executionAdmissionId", maximum=1024),
        "blockDefinitionId": _text(item["blockDefinitionId"], f"{label}.blockDefinitionId", maximum=80),
        "blockDefinitionContentHash": _text(
            item["blockDefinitionContentHash"],
            f"{label}.blockDefinitionContentHash",
            pattern=_BLOCK_DEFINITION_HASH,
        ),
        "blockDefinitionCanonicalSha256": _text(
            item["blockDefinitionCanonicalSha256"],
            f"{label}.blockDefinitionCanonicalSha256",
            pattern=_SHA256,
        ),
        "executionGraphHash": _text(
            item["executionGraphHash"], f"{label}.executionGraphHash", pattern=_BLOCK_GRAPH_HASH
        ),
        "interfaceHash": _text(item["interfaceHash"], f"{label}.interfaceHash", pattern=_BLOCK_INTERFACE_HASH),
    }


def _review(value: Any, label: str) -> dict[str, str]:
    item = _exact_object(value, {"decision", "issuer", "reviewedAt", "notes"}, label)
    if item["decision"] != "semantic_equivalent":
        raise LegacyClusterSemanticEquivalenceError(f"{label}.decision must be semantic_equivalent.")
    reviewed_at = _text(item["reviewedAt"], f"{label}.reviewedAt", maximum=64)
    try:
        parsed = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LegacyClusterSemanticEquivalenceError(f"{label}.reviewedAt must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LegacyClusterSemanticEquivalenceError(f"{label}.reviewedAt must include a UTC offset.")
    return {
        "decision": "semantic_equivalent",
        "issuer": _text(item["issuer"], f"{label}.issuer", maximum=256),
        "reviewedAt": reviewed_at,
        "notes": _text(item["notes"], f"{label}.notes", maximum=4096),
    }


def validate_semantic_equivalence_receipt(value: Any) -> dict[str, Any]:
    receipt = _exact_object(
        value,
        {"id", "historical", "destination", "review", "receiptHash"},
        "Semantic-equivalence receipt",
    )
    normalized = {
        "id": _text(receipt["id"], "Semantic-equivalence receipt.id", pattern=_RECEIPT_ID),
        "historical": _historical_binding(receipt["historical"], "Semantic-equivalence receipt.historical"),
        "destination": _destination_binding(receipt["destination"], "Semantic-equivalence receipt.destination"),
        "review": _review(receipt["review"], "Semantic-equivalence receipt.review"),
        "receiptHash": _text(receipt["receiptHash"], "Semantic-equivalence receipt.receiptHash", pattern=_SHA256),
    }
    expected = canonical_content_hash(normalized, omit="receiptHash")
    if normalized["receiptHash"] != expected:
        raise LegacyClusterSemanticEquivalenceError(
            "Semantic-equivalence receipt hash does not match its complete reviewed body."
        )
    return normalized


def validate_semantic_equivalence_receipts(value: Any) -> dict[str, Any]:
    ledger = _exact_object(
        value,
        {"schemaVersion", "kind", "boundary", "receipts", "contentHash"},
        "Semantic-equivalence receipt ledger",
    )
    if (
        ledger["schemaVersion"] != SEMANTIC_EQUIVALENCE_SCHEMA_VERSION
        or ledger["kind"] != "legacy_cluster_semantic_equivalence_receipts"
        or ledger["boundary"] != _BOUNDARY
    ):
        raise LegacyClusterSemanticEquivalenceError(
            "Semantic-equivalence receipt ledger boundary or version is invalid."
        )
    receipts_value = ledger["receipts"]
    if not isinstance(receipts_value, list) or len(receipts_value) > 512:
        raise LegacyClusterSemanticEquivalenceError("Semantic-equivalence receipt collection is invalid.")
    receipts = [validate_semantic_equivalence_receipt(item) for item in receipts_value]
    if receipts != sorted(receipts, key=lambda item: item["id"]):
        raise LegacyClusterSemanticEquivalenceError(
            "Semantic-equivalence receipts must use canonical receipt ID order."
        )
    receipt_ids = [item["id"] for item in receipts]
    receipt_hashes = [item["receiptHash"] for item in receipts]
    historical_keys = [
        (
            item["historical"]["manifestDefinitionId"],
            item["historical"]["libraryRevision"],
            item["historical"]["manifestContentHash"],
            item["historical"]["executionAdmissionId"],
        )
        for item in receipts
    ]
    if (
        len(receipt_ids) != len(set(receipt_ids))
        or len(receipt_hashes) != len(set(receipt_hashes))
        or len(historical_keys) != len(set(historical_keys))
    ):
        raise LegacyClusterSemanticEquivalenceError("Semantic-equivalence receipts contain duplicate authority.")
    normalized = {
        "schemaVersion": SEMANTIC_EQUIVALENCE_SCHEMA_VERSION,
        "kind": "legacy_cluster_semantic_equivalence_receipts",
        "boundary": deepcopy(_BOUNDARY),
        "receipts": receipts,
        "contentHash": _text(
            ledger["contentHash"], "Semantic-equivalence receipt ledger.contentHash", pattern=_SHA256
        ),
    }
    if normalized["contentHash"] != canonical_content_hash(normalized):
        raise LegacyClusterSemanticEquivalenceError(
            "Semantic-equivalence receipt ledger content hash does not match its complete body."
        )
    return normalized


@lru_cache(maxsize=1)
def _checked_in_semantic_equivalence_receipts() -> dict[str, Any]:
    try:
        raw = SEMANTIC_EQUIVALENCE_RECEIPTS_PATH.read_bytes()
        if len(raw) > _MAX_LEDGER_BYTES:
            raise LegacyClusterSemanticEquivalenceError("Semantic-equivalence receipt ledger is too large.")
        document = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LegacyClusterSemanticEquivalenceError("Semantic-equivalence receipt ledger cannot be loaded.") from exc
    return validate_semantic_equivalence_receipts(document)


def reviewed_semantic_equivalence_receipts() -> dict[str, Any]:
    """Return a detached copy of the checked-in receipt ledger."""

    return deepcopy(_checked_in_semantic_equivalence_receipts())


def semantic_equivalence_receipt_by_reference(receipt_id: str, receipt_hash: str) -> dict[str, Any] | None:
    """Resolve one exact checked-in receipt reference without label inference."""

    matches = [
        receipt
        for receipt in _checked_in_semantic_equivalence_receipts()["receipts"]
        if receipt["id"] == receipt_id and receipt["receiptHash"] == receipt_hash
    ]
    if len(matches) > 1:
        raise LegacyClusterSemanticEquivalenceError("Semantic-equivalence receipt reference is ambiguous.")
    return deepcopy(matches[0]) if matches else None


def semantic_equivalence_receipt_for_historical(
    *,
    manifest_definition_id: str,
    library_revision: str,
    manifest_content_hash: str,
    execution_admission_id: str,
) -> dict[str, Any] | None:
    """Resolve one historical tuple to its exact checked-in destination review."""

    matches = [
        receipt
        for receipt in _checked_in_semantic_equivalence_receipts()["receipts"]
        if receipt["historical"]["manifestDefinitionId"] == manifest_definition_id
        and receipt["historical"]["libraryRevision"] == library_revision
        and receipt["historical"]["manifestContentHash"] == manifest_content_hash
        and receipt["historical"]["executionAdmissionId"] == execution_admission_id
    ]
    if len(matches) > 1:
        raise LegacyClusterSemanticEquivalenceError("Historical semantic-equivalence authority is ambiguous.")
    return deepcopy(matches[0]) if matches else None
