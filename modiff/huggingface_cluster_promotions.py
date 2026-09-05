"""Immutable, route-specific publication receipts for reviewed Cluster outputs.

Technical execution profiles are often shared by several workflow modes.  A
human-approved output therefore promotes only its exact Cluster admission, not
every route that happens to reuse the same loader profile.  This registry is
data-only and never reads the ignored local media or qualification workspace at
runtime.
"""

from __future__ import annotations

from copy import deepcopy
from contextvars import ContextVar
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


CLUSTER_PROMOTION_RECEIPTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "huggingface-cluster-promotion-receipts.v2.json"
)
CLUSTER_PROMOTION_RECEIPTS_SCHEMA_VERSION = 2
_MAX_RECEIPT_BYTES = 256 * 1024
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BLOCK_DEFINITION_HASH = re.compile(r"^block-definition-v2-[0-9a-f]{8}$")
_STUDIO_SPEC_HASH = re.compile(r"^studio-spec-v1-[0-9a-f]{8}$")
_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_MEDIA_KINDS = frozenset({"audio", "image", "video"})
_HISTORICAL_REASON = "legacy_schema_v1_missing_exact_block_definition_identity"
_POST_PROMOTION_PROJECTION_ADMISSION: ContextVar[str | None] = ContextVar(
    "post_promotion_projection_admission",
    default=None,
)
_LIFECYCLE_FIELDS = frozenset(
    {
        "insertedThroughFrontend",
        "expandedOfficialBlocks",
        "editedParameters",
        "savedAndRefreshed",
        "valuesRestoredExactly",
        "collapsedExpandedGraphEquivalent",
        "generatedThroughFrontend",
    }
)
_BOUNDARY = {
    "assetsRemainLocalAndUnpublished": True,
    "approvalIsRouteSpecific": True,
    "autoEligibilityRemainsResourceGated": True,
    "executableRemainsRuntimeGated": True,
    "exactBlockDefinitionBindingRequired": True,
    "legacyApprovalsAreHistoricalOnly": True,
}


class ClusterPromotionReceiptError(ValueError):
    """Raised when checked-in Cluster publication evidence is malformed or stale."""


def canonical_content_hash(document: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(document))
    payload.pop("contentHash", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _require_exact_keys(value: Any, keys: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ClusterPromotionReceiptError(f"{label} has missing or unknown fields.")
    return value


def _validate_receipt_collection(
    receipts: Any,
    *,
    require_block_definition: bool,
    label: str,
) -> None:
    if not isinstance(receipts, list) or len(receipts) > 256:
        raise ClusterPromotionReceiptError(f"{label} collection is invalid.")
    if receipts != sorted(receipts, key=lambda item: str(item.get("id", ""))):
        raise ClusterPromotionReceiptError(f"{label} must use canonical ID order.")
    seen_ids: set[str] = set()
    seen_admissions: set[str] = set()
    seen_tasks: set[str] = set()
    for receipt in receipts:
        receipt_keys = {
            "id",
            "definitionId",
            "admissionId",
            "artifact",
            "studioExecutionSpec",
            "evidence",
            "review",
            "publication",
        }
        if require_block_definition:
            receipt_keys.add("blockDefinition")
        receipt = _require_exact_keys(
            receipt,
            receipt_keys,
            label=label.rstrip("s"),
        )
        receipt_id = receipt["id"]
        admission_id = receipt["admissionId"]
        if (
            not isinstance(receipt_id, str)
            or not receipt_id.startswith("cluster-promotion:")
            or receipt_id in seen_ids
            or not isinstance(receipt["definitionId"], str)
            or not receipt["definitionId"].startswith("diffusers.")
            or not isinstance(admission_id, str)
            or not admission_id.startswith("diffusers.cluster-admission:")
            or admission_id in seen_admissions
        ):
            raise ClusterPromotionReceiptError("Cluster promotion receipt identity is invalid or duplicated.")
        seen_ids.add(receipt_id)
        seen_admissions.add(admission_id)

        if require_block_definition:
            block_definition = _require_exact_keys(
                receipt["blockDefinition"],
                {"definitionId", "contentHash", "canonicalSha256"},
                label="BlockDefinitionV2 binding",
            )
            if (
                block_definition["definitionId"] != admission_id
                or not _BLOCK_DEFINITION_HASH.fullmatch(str(block_definition["contentHash"]))
                or not str(block_definition["canonicalSha256"]).startswith("sha256:")
                or not _SHA256.fullmatch(str(block_definition["canonicalSha256"])[7:])
            ):
                raise ClusterPromotionReceiptError(
                    "Cluster promotion BlockDefinitionV2 binding is invalid."
                )

        artifact = _require_exact_keys(receipt["artifact"], {"repository", "revision"}, label="Artifact binding")
        if (
            not isinstance(artifact["repository"], str)
            or "/" not in artifact["repository"]
            or not _COMMIT.fullmatch(str(artifact["revision"]))
        ):
            raise ClusterPromotionReceiptError("Cluster promotion artifact binding is invalid.")

        studio_spec = _require_exact_keys(
            receipt["studioExecutionSpec"],
            {"id", "contentHash", "executionProfileId"},
            label="Studio execution binding",
        )
        if (
            not isinstance(studio_spec["id"], str)
            or not isinstance(studio_spec["executionProfileId"], str)
            or not _STUDIO_SPEC_HASH.fullmatch(str(studio_spec["contentHash"]))
        ):
            raise ClusterPromotionReceiptError("Cluster promotion Studio execution binding is invalid.")

        evidence = _require_exact_keys(
            receipt["evidence"],
            {
                "taskId",
                "runtimeFingerprint",
                "mediaKind",
                "mediaSha256",
                "mediaByteSize",
                "lifecycle",
            },
            label="Promotion evidence",
        )
        task_id = evidence["taskId"]
        if (
            not _TASK_ID.fullmatch(str(task_id))
            or task_id in seen_tasks
            or not str(evidence["runtimeFingerprint"]).startswith("sha256:")
            or not _SHA256.fullmatch(str(evidence["runtimeFingerprint"])[7:])
            or evidence["mediaKind"] not in _MEDIA_KINDS
            or not _SHA256.fullmatch(str(evidence["mediaSha256"]))
            or isinstance(evidence["mediaByteSize"], bool)
            or not isinstance(evidence["mediaByteSize"], int)
            or evidence["mediaByteSize"] <= 0
        ):
            raise ClusterPromotionReceiptError("Cluster promotion execution evidence is invalid.")
        seen_tasks.add(task_id)
        lifecycle = _require_exact_keys(
            evidence["lifecycle"],
            set(_LIFECYCLE_FIELDS),
            label="Promotion lifecycle",
        )
        if any(value is not True for value in lifecycle.values()):
            raise ClusterPromotionReceiptError("A promoted Cluster route requires every lifecycle proof.")

        review = _require_exact_keys(
            receipt["review"],
            {"decision", "reviewedAt", "reviewerRole", "comment"},
            label="Promotion review",
        )
        reviewed_at = review["reviewedAt"]
        try:
            parsed_reviewed_at = datetime.fromisoformat(
                str(reviewed_at).replace("Z", "+00:00")
            )
        except ValueError:
            parsed_reviewed_at = None
        if (
            review["decision"] != "approved"
            or review["reviewerRole"] != "workspace_owner"
            or not isinstance(reviewed_at, str)
            or parsed_reviewed_at is None
            or parsed_reviewed_at.tzinfo is None
            or parsed_reviewed_at.utcoffset() is None
            or not isinstance(review["comment"], str)
            or not review["comment"].strip()
        ):
            raise ClusterPromotionReceiptError("Cluster promotion requires an explicit workspace-owner approval.")
        if receipt["publication"] != {
            "liveProof": True,
            "executable": False,
            "autoEligible": False,
            "galleryEligible": False,
        }:
            raise ClusterPromotionReceiptError("Cluster promotion receipt weakens the runtime publication boundary.")


def validate_cluster_promotion_receipts(document: Any) -> dict[str, Any]:
    document = _require_exact_keys(
        document,
        {
            "schemaVersion",
            "kind",
            "boundary",
            "receipts",
            "historicalReceipts",
            "contentHash",
        },
        label="Cluster promotion receipt ledger",
    )
    if (
        document["schemaVersion"] != CLUSTER_PROMOTION_RECEIPTS_SCHEMA_VERSION
        or document["kind"] != "huggingface_cluster_route_promotions"
        or document["boundary"] != _BOUNDARY
        or document["contentHash"] != canonical_content_hash(document)
    ):
        raise ClusterPromotionReceiptError("Cluster promotion receipt ledger boundary or hash is invalid.")

    _validate_receipt_collection(
        document["receipts"],
        require_block_definition=True,
        label="Cluster promotion receipts",
    )
    historical = document["historicalReceipts"]
    if not isinstance(historical, list) or len(historical) > 256:
        raise ClusterPromotionReceiptError("Historical Cluster promotion receipt collection is invalid.")
    historical_receipts = []
    for record in historical:
        record = _require_exact_keys(
            record,
            {"legacySchemaVersion", "nonAuthorizingReason", "receipt"},
            label="Historical Cluster promotion receipt",
        )
        if (
            record["legacySchemaVersion"] != 1
            or record["nonAuthorizingReason"] != _HISTORICAL_REASON
        ):
            raise ClusterPromotionReceiptError(
                "Historical Cluster promotion receipt authority boundary is invalid."
            )
        historical_receipts.append(record["receipt"])
    _validate_receipt_collection(
        historical_receipts,
        require_block_definition=False,
        label="Historical Cluster promotion receipts",
    )
    return deepcopy(document)


@lru_cache(maxsize=1)
def _reviewed_cluster_promotion_receipts() -> dict[str, Any]:
    try:
        encoded = CLUSTER_PROMOTION_RECEIPTS_PATH.read_bytes()
        if len(encoded) > _MAX_RECEIPT_BYTES:
            raise ClusterPromotionReceiptError("Cluster promotion receipt ledger is too large.")
        document = json.loads(encoded)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ClusterPromotionReceiptError("Cluster promotion receipt ledger cannot be loaded.") from error
    return validate_cluster_promotion_receipts(document)


def reviewed_cluster_promotion_receipts() -> dict[str, Any]:
    """Return current receipts plus detached, explicitly historical approvals."""

    return deepcopy(_reviewed_cluster_promotion_receipts())


def historical_promotion_receipt_for_admission(admission_id: str) -> dict[str, Any] | None:
    """Return preserved legacy review evidence without granting authority."""

    matches = [
        record["receipt"]
        for record in _reviewed_cluster_promotion_receipts()["historicalReceipts"]
        if record["receipt"]["admissionId"] == admission_id
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ClusterPromotionReceiptError("Cluster admission has duplicate historical promotion receipts.")
    return deepcopy(matches[0])


def build_post_promotion_node_library_candidate(admission_id: str) -> dict[str, Any]:
    """Project one liveProof=true catalog candidate without installing authority."""

    from modiff.huggingface_node_library import (
        build_huggingface_node_library,
        reviewed_huggingface_node_library,
    )

    current = reviewed_huggingface_node_library()
    current_admissions = [
        admission
        for definition in current["definitions"]
        for admission in definition.get("executionAdmissions", ())
        if admission.get("id") == admission_id
    ]
    if len(current_admissions) != 1:
        raise ClusterPromotionReceiptError(
            "Promotion admission does not resolve to one current catalog route."
        )
    if current_admissions[0].get("publication", {}).get("liveProof") is not False:
        raise ClusterPromotionReceiptError(
            "Post-promotion projection requires a currently unpromoted admission."
        )
    token = _POST_PROMOTION_PROJECTION_ADMISSION.set(admission_id)
    try:
        return build_huggingface_node_library()
    finally:
        _POST_PROMOTION_PROJECTION_ADMISSION.reset(token)


def build_cluster_promotion_receipt_candidate(
    *,
    admission_id: str,
    provenance: Mapping[str, Any],
    media_sha256: str,
    media_byte_size: int,
    receipt_id: str,
    reviewed_at: str,
    review_comment: str,
    lifecycle_confirmations: Mapping[str, Any],
    workspace_owner_approved: bool,
    post_promotion_block_definition: Mapping[str, Any],
    post_promotion_manifest_content_hash: str,
    output_index: int = 0,
) -> dict[str, Any]:
    """Build, but never approve or install, one exact schema-v2 receipt candidate.

    Every human assertion is an explicit argument.  The function derives only
    machine-verifiable route, runtime, task, and media bindings from a current
    frontend provenance document and the checked-in catalog.  Callers must
    separately review and check in the returned candidate.
    """

    if workspace_owner_approved is not True:
        raise ClusterPromotionReceiptError(
            "Promotion receipt generation requires explicit workspace-owner approval."
        )
    lifecycle = _require_exact_keys(
        dict(lifecycle_confirmations),
        set(_LIFECYCLE_FIELDS),
        label="Promotion lifecycle confirmations",
    )
    if any(value is not True for value in lifecycle.values()):
        raise ClusterPromotionReceiptError(
            "Promotion receipt generation requires every lifecycle confirmation."
        )
    if (
        not isinstance(media_sha256, str)
        or not _SHA256.fullmatch(media_sha256)
        or isinstance(media_byte_size, bool)
        or not isinstance(media_byte_size, int)
        or media_byte_size <= 0
    ):
        raise ClusterPromotionReceiptError("Promotion media evidence is empty or invalid.")
    if isinstance(output_index, bool) or not isinstance(output_index, int) or output_index < 0:
        raise ClusterPromotionReceiptError("Promotion output index is invalid.")

    # Imported lazily to avoid the promotion -> catalog -> promotion import
    # cycle during normal node-library construction.
    from modiff.huggingface_cluster_runtime import REGISTERED_BLOCK_V2_DEFINITION_PINS
    from modiff.huggingface_node_library import reviewed_huggingface_node_library

    matches = [
        (definition, admission)
        for definition in reviewed_huggingface_node_library()["definitions"]
        for admission in definition.get("executionAdmissions", ())
        if admission.get("id") == admission_id
    ]
    if len(matches) != 1:
        raise ClusterPromotionReceiptError(
            "Promotion admission does not resolve to one current catalog route."
        )
    definition, admission = matches[0]
    current_definition_id = definition["id"]
    block_pin = REGISTERED_BLOCK_V2_DEFINITION_PINS.get(admission_id)
    if block_pin is None:
        raise ClusterPromotionReceiptError(
            "Promotion admission has no current registered BlockDefinitionV2 pin."
        )
    expected_block = {
        "definitionId": admission_id,
        "contentHash": block_pin[0],
        "canonicalSha256": block_pin[1],
    }
    expected_spec = deepcopy(admission["studioExecutionSpec"])
    expected_artifact = {
        "repository": admission["artifact"]["repo"],
        "revision": admission["artifact"]["revision"],
    }
    post_promotion_library = build_post_promotion_node_library_candidate(admission_id)
    post_promotion_definition = next(
        candidate_definition
        for candidate_definition in post_promotion_library["definitions"]
        if candidate_definition["id"] == current_definition_id
        and any(
            item.get("id") == admission_id
            for item in candidate_definition.get("executionAdmissions", ())
        )
    )
    promoted_block = _require_exact_keys(
        dict(post_promotion_block_definition),
        {"definitionId", "contentHash", "canonicalSha256"},
        label="Post-promotion BlockDefinitionV2 binding",
    )
    if (
        post_promotion_manifest_content_hash != post_promotion_definition["contentHash"]
        or promoted_block.get("definitionId") != admission_id
        or not _BLOCK_DEFINITION_HASH.fullmatch(str(promoted_block.get("contentHash")))
        or not str(promoted_block.get("canonicalSha256", "")).startswith("sha256:")
        or not _SHA256.fullmatch(str(promoted_block.get("canonicalSha256", ""))[7:])
    ):
        raise ClusterPromotionReceiptError(
            "Post-promotion compiler identity does not match the projected current catalog."
        )

    if (
        provenance.get("schemaVersion") != 2
        or provenance.get("format") != "modiff.live-proof.provenance.v2"
        or provenance.get("blockers") != []
    ):
        raise ClusterPromotionReceiptError(
            "Promotion evidence is not one complete, unblocked frontend provenance document."
        )
    route_binding = provenance.get("routeBinding")
    if not isinstance(route_binding, Mapping) or (
        route_binding.get("admissionId") != admission_id
        or route_binding.get("blockDefinition") != expected_block
        or route_binding.get("studioExecutionSpec") != expected_spec
        or route_binding.get("artifact") != expected_artifact
    ):
        raise ClusterPromotionReceiptError(
            "Promotion evidence is stale for the current exact route binding."
        )

    outputs = provenance.get("output")
    items = outputs.get("items") if isinstance(outputs, Mapping) else None
    if not isinstance(items, list) or output_index >= len(items):
        raise ClusterPromotionReceiptError("Promotion evidence output is missing.")
    output = items[output_index]
    if not isinstance(output, Mapping):
        raise ClusterPromotionReceiptError("Promotion evidence output is invalid.")
    if (
        output.get("index") != output_index
        or output.get("mediaType") not in _MEDIA_KINDS
        or output.get("byteSize") != media_byte_size
        or output.get("encodedSha256") != f"sha256:encoded:{media_sha256}"
    ):
        raise ClusterPromotionReceiptError(
            "Promotion media bytes do not match the selected frontend output."
        )
    runtime_fingerprint = provenance.get("backendReportedRuntimeFingerprint")
    task_id = provenance.get("taskId")
    if (
        not _TASK_ID.fullmatch(str(task_id))
        or not isinstance(runtime_fingerprint, str)
        or not runtime_fingerprint.startswith("sha256:")
        or not _SHA256.fullmatch(runtime_fingerprint[7:])
    ):
        raise ClusterPromotionReceiptError(
            "Promotion evidence task or backend runtime fingerprint is invalid."
        )

    receipt = {
        "id": receipt_id,
        "definitionId": definition["id"],
        "admissionId": admission_id,
        "blockDefinition": deepcopy(promoted_block),
        "artifact": expected_artifact,
        "studioExecutionSpec": expected_spec,
        "evidence": {
            "taskId": task_id,
            "runtimeFingerprint": runtime_fingerprint,
            "mediaKind": output["mediaType"],
            "mediaSha256": media_sha256,
            "mediaByteSize": media_byte_size,
            "lifecycle": deepcopy(dict(lifecycle)),
        },
        "review": {
            "decision": "approved",
            "reviewedAt": reviewed_at,
            "reviewerRole": "workspace_owner",
            "comment": review_comment,
        },
        "publication": {
            "liveProof": True,
            "executable": False,
            "autoEligible": False,
            "galleryEligible": False,
        },
    }
    _validate_receipt_collection(
        [receipt],
        require_block_definition=True,
        label="Cluster promotion receipt candidates",
    )
    return deepcopy(receipt)


def build_cluster_promotion_ledger_candidate(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated ledger candidate without writing or granting approval."""

    ledger = reviewed_cluster_promotion_receipts()
    if any(
        current.get("admissionId") == receipt.get("admissionId")
        for current in ledger["receipts"]
    ):
        raise ClusterPromotionReceiptError(
            "A current receipt already exists for this admission; replacement requires explicit review."
        )
    ledger["receipts"] = sorted(
        [*ledger["receipts"], deepcopy(dict(receipt))],
        key=lambda item: item["id"],
    )
    ledger["contentHash"] = canonical_content_hash(ledger)
    return validate_cluster_promotion_receipts(ledger)


def promotion_receipt_for_admission(
    admission_id: str,
    *,
    definition_id: str,
    artifact: Mapping[str, Any],
    studio_execution_spec: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Resolve a promotion only when every immutable execution binding matches."""

    if _POST_PROMOTION_PROJECTION_ADMISSION.get() == admission_id:
        # This context-local sentinel exists only while constructing the
        # non-authorizing post-promotion catalog candidate. It is never loaded
        # from disk, returned by the public ledger API, or treated as approval.
        return {"projectionOnly": True}

    matches = [
        receipt
        for receipt in _reviewed_cluster_promotion_receipts()["receipts"]
        if receipt["admissionId"] == admission_id
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ClusterPromotionReceiptError("Cluster admission has duplicate promotion receipts.")
    receipt = matches[0]
    # Import lazily so static receipt parsing stays data-only and so the
    # promotion module does not create a library/runtime import cycle.  The
    # checked-in runtime pin is the backend authority for the exact current
    # BlockDefinitionV2 identity.
    from modiff.huggingface_cluster_runtime import REGISTERED_BLOCK_V2_DEFINITION_PINS

    block_pin = REGISTERED_BLOCK_V2_DEFINITION_PINS.get(admission_id)
    expected_block_definition = (
        {
            "definitionId": admission_id,
            "contentHash": block_pin[0],
            "canonicalSha256": block_pin[1],
        }
        if block_pin is not None
        else None
    )
    expected_artifact = {
        "repository": artifact.get("repo"),
        "revision": artifact.get("revision"),
    }
    expected_spec = {
        "id": studio_execution_spec.get("id"),
        "contentHash": studio_execution_spec.get("contentHash"),
        "executionProfileId": studio_execution_spec.get("executionProfileId"),
    }
    if (
        receipt["definitionId"] != definition_id
        or expected_block_definition is None
        or receipt["blockDefinition"] != expected_block_definition
        or receipt["artifact"] != expected_artifact
        or receipt["studioExecutionSpec"] != expected_spec
    ):
        raise ClusterPromotionReceiptError(
            "Cluster promotion receipt is stale for the current definition, artifact, or execution graph."
        )
    return deepcopy(receipt)
