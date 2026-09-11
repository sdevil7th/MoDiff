"""Reviewed compiler destinations for exact archived Cluster definitions.

A mapping binds one collision-resistant archived definition/admission/Studio
tuple to one exact registered ``BlockDefinitionV2`` destination.  It is not a
semantic-equivalence assertion and it does not mutate a workflow.  The normal
compiler supplement must still preserve every instance value, port, preview,
projection node, edge, and layout, and apply still requires explicit review,
byte backup, and rollback support.
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

from modiff.legacy_cluster_archived_definitions import (
    reviewed_archived_cluster_definition_evidence,
)


HISTORICAL_COMPILER_MAPPINGS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "legacy-cluster-compiler-mappings.v1.json"
)
HISTORICAL_COMPILER_MAPPING_SCHEMA_VERSION = 1
_MAX_LEDGER_BYTES = 2 * 1024 * 1024
_MAX_MAPPINGS = 256

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_STUDIO_HASH = re.compile(r"^studio-spec-v1-[0-9a-f]{8}$")
_BLOCK_DEFINITION_HASH = re.compile(r"^block-definition-v2-[0-9a-f]{8}$")
_BLOCK_GRAPH_HASH = re.compile(r"^block-graph-v2-[0-9a-f]{8}$")
_BLOCK_INTERFACE_HASH = re.compile(r"^block-interface-v2-[0-9a-f]{8}$")
_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,1023}$")
_MAPPING_ID = re.compile(r"^legacy-cluster-compiler-mapping:[A-Za-z0-9._:-]{1,180}$")

_BOUNDARY = {
    "checkedInCompilerMappingAuthority": True,
    "requiresExactArchivedDefinition": True,
    "doesNotAssertSemanticEquivalence": True,
    "doesNotInferFromLabelsOrModels": True,
    "requiresPerInstancePreservationReceipts": True,
    "doesNotAuthorizeWorkflowMutation": True,
}


class LegacyClusterCompilerMappingError(ValueError):
    """Raised when historical compiler-mapping authority is malformed."""


def canonical_content_hash(value: Mapping[str, Any], *, omit: str = "contentHash") -> str:
    body = deepcopy(dict(value))
    body.pop(omit, None)
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _exact_object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise LegacyClusterCompilerMappingError(f"{label} has missing or unknown fields.")
    return value


def _text(
    value: Any,
    label: str,
    *,
    maximum: int = 4096,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or (pattern is not None and pattern.fullmatch(value) is None)
    ):
        raise LegacyClusterCompilerMappingError(f"{label} is invalid.")
    return value


def _studio_spec(value: Any, label: str) -> dict[str, str]:
    item = _exact_object(value, {"id", "contentHash", "executionProfileId"}, label)
    return {
        "id": _text(item["id"], f"{label}.id", maximum=512, pattern=_PUBLIC_ID),
        "contentHash": _text(item["contentHash"], f"{label}.contentHash", pattern=_STUDIO_HASH),
        "executionProfileId": _text(
            item["executionProfileId"],
            f"{label}.executionProfileId",
            maximum=512,
            pattern=_PUBLIC_ID,
        ),
    }


def _historical(value: Any, label: str) -> dict[str, Any]:
    item = _exact_object(
        value,
        {
            "manifestDefinitionId",
            "libraryRevision",
            "manifestContentHash",
            "executionAdmissionId",
            "studioExecutionSpec",
            "archivedDefinitionRecordHash",
            "blockContractHash",
            "rootBlockDefinitionId",
        },
        label,
    )
    return {
        "manifestDefinitionId": _text(
            item["manifestDefinitionId"], f"{label}.manifestDefinitionId", pattern=_PUBLIC_ID
        ),
        "libraryRevision": _text(
            item["libraryRevision"], f"{label}.libraryRevision", pattern=_COMMIT
        ),
        "manifestContentHash": _text(
            item["manifestContentHash"], f"{label}.manifestContentHash", pattern=_SHA256
        ),
        "executionAdmissionId": _text(
            item["executionAdmissionId"], f"{label}.executionAdmissionId", pattern=_PUBLIC_ID
        ),
        "studioExecutionSpec": _studio_spec(
            item["studioExecutionSpec"], f"{label}.studioExecutionSpec"
        ),
        "archivedDefinitionRecordHash": _text(
            item["archivedDefinitionRecordHash"],
            f"{label}.archivedDefinitionRecordHash",
            pattern=_SHA256,
        ),
        "blockContractHash": _text(
            item["blockContractHash"], f"{label}.blockContractHash", pattern=_SHA256
        ),
        "rootBlockDefinitionId": _text(
            item["rootBlockDefinitionId"], f"{label}.rootBlockDefinitionId", maximum=1024
        ),
    }


def _destination(value: Any, label: str) -> dict[str, str]:
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
        "manifestDefinitionId": _text(
            item["manifestDefinitionId"], f"{label}.manifestDefinitionId", pattern=_PUBLIC_ID
        ),
        "libraryRevision": _text(item["libraryRevision"], f"{label}.libraryRevision", pattern=_COMMIT),
        "manifestContentHash": _text(
            item["manifestContentHash"], f"{label}.manifestContentHash", pattern=_SHA256
        ),
        "executionAdmissionId": _text(
            item["executionAdmissionId"], f"{label}.executionAdmissionId", pattern=_PUBLIC_ID
        ),
        "blockDefinitionId": _text(
            item["blockDefinitionId"], f"{label}.blockDefinitionId", maximum=384
        ),
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
        "interfaceHash": _text(
            item["interfaceHash"], f"{label}.interfaceHash", pattern=_BLOCK_INTERFACE_HASH
        ),
    }


def _review(value: Any, label: str) -> dict[str, str]:
    item = _exact_object(value, {"decision", "issuer", "reviewedAt", "notes"}, label)
    if item["decision"] != "historical_compiler_mapping_reviewed":
        raise LegacyClusterCompilerMappingError(
            f"{label}.decision must be historical_compiler_mapping_reviewed."
        )
    reviewed_at = _text(item["reviewedAt"], f"{label}.reviewedAt", maximum=64)
    try:
        parsed = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LegacyClusterCompilerMappingError(f"{label}.reviewedAt must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LegacyClusterCompilerMappingError(f"{label}.reviewedAt must include a UTC offset.")
    return {
        "decision": "historical_compiler_mapping_reviewed",
        "issuer": _text(item["issuer"], f"{label}.issuer", maximum=256),
        "reviewedAt": reviewed_at,
        "notes": _text(item["notes"], f"{label}.notes", maximum=4096),
    }


def _archive_records(archive_ledger: Mapping[str, Any]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in archive_ledger.get("records", []):
        if not isinstance(record, Mapping):
            continue
        definition = record.get("definition")
        registration = record.get("registration")
        if not isinstance(definition, Mapping) or not isinstance(registration, Mapping):
            continue
        if registration.get("status") != "recovery_audit_registered":
            continue
        for admission in definition.get("executionAdmissions", []):
            if not isinstance(admission, Mapping):
                continue
            key = (
                str(definition.get("id")),
                str(definition.get("libraryRevision")),
                str(definition.get("contentHash")),
                str(admission.get("id")),
            )
            if key in result:
                raise LegacyClusterCompilerMappingError(
                    "Archived definition evidence contains duplicate execution authority."
                )
            result[key] = {"record": record, "definition": definition, "admission": admission}
    return result


def validate_historical_compiler_mapping(
    value: Any,
    *,
    archive_ledger: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    mapping = _exact_object(
        value,
        {"id", "historical", "destination", "review", "mappingHash"},
        "Historical compiler mapping",
    )
    normalized = {
        "id": _text(mapping["id"], "Historical compiler mapping.id", pattern=_MAPPING_ID),
        "historical": _historical(mapping["historical"], "Historical compiler mapping.historical"),
        "destination": _destination(mapping["destination"], "Historical compiler mapping.destination"),
        "review": _review(mapping["review"], "Historical compiler mapping.review"),
        "mappingHash": _text(
            mapping["mappingHash"], "Historical compiler mapping.mappingHash", pattern=_SHA256
        ),
    }
    if normalized["mappingHash"] != canonical_content_hash(normalized, omit="mappingHash"):
        raise LegacyClusterCompilerMappingError(
            "Historical compiler mapping hash does not match its complete body."
        )
    archive = archive_ledger or reviewed_archived_cluster_definition_evidence()
    historical = normalized["historical"]
    key = (
        historical["manifestDefinitionId"],
        historical["libraryRevision"],
        historical["manifestContentHash"],
        historical["executionAdmissionId"],
    )
    evidence = _archive_records(archive).get(key)
    if evidence is None:
        raise LegacyClusterCompilerMappingError(
            "Historical compiler mapping does not resolve to one registered exact archived definition admission."
        )
    record = evidence["record"]
    definition = evidence["definition"]
    admission = evidence["admission"]
    expected = {
        "manifestDefinitionId": definition["id"],
        "libraryRevision": definition["libraryRevision"],
        "manifestContentHash": definition["contentHash"],
        "executionAdmissionId": admission["id"],
        "studioExecutionSpec": admission["studioExecutionSpec"],
        "archivedDefinitionRecordHash": record["recordHash"],
        "blockContractHash": definition["blockContractHash"],
        "rootBlockDefinitionId": definition["rootBlockDefinitionId"],
    }
    if historical != expected:
        raise LegacyClusterCompilerMappingError(
            "Historical compiler mapping source does not match the exact archived definition body."
        )
    return normalized


def validate_historical_compiler_mappings(
    value: Any,
    *,
    archive_ledger: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ledger = _exact_object(
        value,
        {"schemaVersion", "kind", "boundary", "mappings", "contentHash"},
        "Historical compiler mapping ledger",
    )
    if (
        ledger["schemaVersion"] != HISTORICAL_COMPILER_MAPPING_SCHEMA_VERSION
        or ledger["kind"] != "legacy_cluster_compiler_mappings"
        or ledger["boundary"] != _BOUNDARY
    ):
        raise LegacyClusterCompilerMappingError(
            "Historical compiler mapping ledger boundary or version is invalid."
        )
    raw_mappings = ledger["mappings"]
    if not isinstance(raw_mappings, list) or len(raw_mappings) > _MAX_MAPPINGS:
        raise LegacyClusterCompilerMappingError("Historical compiler mapping collection is invalid.")
    mappings = [
        validate_historical_compiler_mapping(item, archive_ledger=archive_ledger)
        for item in raw_mappings
    ]
    if mappings != sorted(mappings, key=lambda item: item["id"]):
        raise LegacyClusterCompilerMappingError(
            "Historical compiler mappings must use canonical mapping ID order."
        )
    ids = [item["id"] for item in mappings]
    hashes = [item["mappingHash"] for item in mappings]
    historical_keys = [
        (
            item["historical"]["manifestDefinitionId"],
            item["historical"]["libraryRevision"],
            item["historical"]["manifestContentHash"],
            item["historical"]["executionAdmissionId"],
        )
        for item in mappings
    ]
    if (
        len(ids) != len(set(ids))
        or len(hashes) != len(set(hashes))
        or len(historical_keys) != len(set(historical_keys))
    ):
        raise LegacyClusterCompilerMappingError(
            "Historical compiler mappings contain duplicate authority."
        )
    normalized = {
        "schemaVersion": HISTORICAL_COMPILER_MAPPING_SCHEMA_VERSION,
        "kind": "legacy_cluster_compiler_mappings",
        "boundary": deepcopy(_BOUNDARY),
        "mappings": mappings,
        "contentHash": _text(
            ledger["contentHash"], "Historical compiler mapping ledger.contentHash", pattern=_SHA256
        ),
    }
    if normalized["contentHash"] != canonical_content_hash(normalized):
        raise LegacyClusterCompilerMappingError(
            "Historical compiler mapping ledger content hash does not match its complete body."
        )
    return normalized


@lru_cache(maxsize=1)
def _checked_in_historical_compiler_mappings() -> dict[str, Any]:
    try:
        if HISTORICAL_COMPILER_MAPPINGS_PATH.is_symlink():
            raise LegacyClusterCompilerMappingError(
                "Historical compiler mapping ledger must not be a symlink."
            )
        raw = HISTORICAL_COMPILER_MAPPINGS_PATH.read_bytes()
        if len(raw) > _MAX_LEDGER_BYTES:
            raise LegacyClusterCompilerMappingError("Historical compiler mapping ledger is too large.")
        document = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LegacyClusterCompilerMappingError(
            "Historical compiler mapping ledger cannot be loaded."
        ) from exc
    return validate_historical_compiler_mappings(document)


def reviewed_historical_compiler_mappings() -> dict[str, Any]:
    return deepcopy(_checked_in_historical_compiler_mappings())


def historical_compiler_mapping_by_reference(
    mapping_id: str, mapping_hash: str
) -> dict[str, Any] | None:
    matches = [
        item
        for item in _checked_in_historical_compiler_mappings()["mappings"]
        if item["id"] == mapping_id and item["mappingHash"] == mapping_hash
    ]
    if len(matches) > 1:
        raise LegacyClusterCompilerMappingError(
            "Historical compiler mapping reference is ambiguous."
        )
    return deepcopy(matches[0]) if matches else None


def historical_compiler_mapping_for_historical(
    *,
    manifest_definition_id: str,
    library_revision: str,
    manifest_content_hash: str,
    execution_admission_id: str,
) -> dict[str, Any] | None:
    matches = [
        item
        for item in _checked_in_historical_compiler_mappings()["mappings"]
        if item["historical"]["manifestDefinitionId"] == manifest_definition_id
        and item["historical"]["libraryRevision"] == library_revision
        and item["historical"]["manifestContentHash"] == manifest_content_hash
        and item["historical"]["executionAdmissionId"] == execution_admission_id
    ]
    if len(matches) > 1:
        raise LegacyClusterCompilerMappingError("Historical compiler mapping is ambiguous.")
    return deepcopy(matches[0]) if matches else None
